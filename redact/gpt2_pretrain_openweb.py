'''
GPT2 model pretraining on openwebtext dataset.
References:
Andrej Karpathy's nanoGPT repository: 
https://github.com/karpathy/nanoGPT/blob/924a0873ebf4a3a12437633f5ab9c9fe3d421084/train.py#L135
'''

import argparse
import os
import time
from datetime import datetime
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from gpt2_model import GPTConfig, GPT
from data.custom_datasets.owt_dataset import OWTDataset
from torch.utils.data import DataLoader

dttm = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description=__doc__)
# I/O
parser.add_argument('--out-dir', default='/media/nvme2/models/redact', type=str, help='Output directory for saving checkpoints.')
parser.add_argument('--eval-interval', default=2000, type=int, help='Number of iterations between evaluations.')
parser.add_argument('--log-interval', default=1, type=int, help='Number of iterations between log entries.')
parser.add_argument('--eval-iters', default=200, type=int, help='Number of iterations to perform for each evaluation.')
parser.add_argument('--eval-only', action='store_true', help='Default = false. If true, script exits after the first evaluation.')
parser.add_argument('--always-save-checkpoint', action='store_true', help='Default = false. If true, save checkpoint after each evaluation.  Otherwise, save only after beating best validation loss.')
parser.add_argument('--init-from', default='scratch', type=str, help='Model initialization method.  Either \'scratch\', \'resume\', or \'gpt2...\'.')
# Logging
parser.add_argument( '--wandb-log', action='store_true', help='Default = false.  If true, log results using wandb account.')
parser.add_argument('--wandb-project', default='gpt2_pretraining', type=str, help='Wandb project name.')
parser.add_argument('--wandb-run-name', default=dttm, type=str, help='Wandb run name.')
# Data
parser.add_argument('--dataset', default='openwebtext', type=str, help='Name of training dataset.')
parser.add_argument('--data-dir', default='/media/nvme2/openwebtext', type=str, help='Data directory for training dataset.')
parser.add_argument('--gradient-accumulation-steps', default=5, type=int, help='Used to simulate larger batch sizes.')
parser.add_argument('--batch-size', default=6, type=int, help='Batch size.  If gradient-accumulation-steps > 1, this is the micro-batch size.')
parser.add_argument('--block-size', default=512, type=int, help='Number of input tokens.')
# Model
parser.add_argument('--n-layer', default=12, type=int, help='Number of transformer blocks in model.')
parser.add_argument('--n-head', default=12, type=int, help='Number of attention heads in each transformer block.')
parser.add_argument('--n-embd', default=768, type=int, help='Length of embedding vectors (token & position).')
parser.add_argument('--dropout', default=0.0, type=float, help='For pretraining, 0 is good, for finetuning try 0.1+.')
parser.add_argument('--bias', action='store_true', help='Default = false.  If true, use bias in LayerNorm and Linear layers.')
# Optimizer
parser.add_argument('--learning-rate', default=6e-4, type=float, help='Maximum learning rate.')
parser.add_argument('--max-iters', default=600000, type=int, help='Total number of training iterations.')
parser.add_argument('--weight-decay', default=1e-2, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.95, type=float)
parser.add_argument('--grad-clip', default=1.0, type=float, help='Clip gradients at this value, or disable if = 0.0.')
# Learning rate decay
parser.add_argument('--decay-lr', action='store_false', help='Default = true. If false, do not decay learning rate.')
parser.add_argument('--warmup-iters', default=2000, type=int)
parser.add_argument('--lr-decay-iters', default=600000, type=int, help='Should be ~ max-iters.')
parser.add_argument('--min-lr', default=6e-5, type=float, help='Should be ~ learning_rate/10.')
# DDP settings
parser.add_argument('--backend', default='nccl', type=str, help='\'nccl\', \'gloo\', etc.')
parser.add_argument('--device', default='cuda', type=str, help='\'cpu\', \'cuda\', \'cuda:0\', \'cuda:1\', etc.')
parser.add_argument('--dtype', default='bfloat16', type=str, help='\'float32\', \'bfloat16\', or \'float16\'. The latter will auto-implement a GradScaler.')
parser.add_argument('--compile', action='store_false', help='Default = true.  If false, do not use PyTorch 2.0 to compile the model to be faster.')

args = parser.parse_args()
config = vars(args)

# Set up distributed training
ddp = int(os.environ.get('RANK', -1)) != -1 # Is this a ddp run?
if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    master_process = ddp_rank == 0 # This process will do logging, checkpointing etc.
    seed_offset = ddp_rank # Each process gets a different seed
else:
    # Single gpu, single process
    master_process = True
    device = args.device
    seed_offset = 0

if master_process:
    os.makedirs(args.out_dir, exist_ok=True)

torch.manual_seed(421 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # Allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # Allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # For later us in torch.autocast
# Note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type)

# Pytorch data loaders
owt_train_ds = OWTDataset(
    dir=args.data_dir, split='train', block_size=args.block_size
)
owt_val_ds = OWTDataset(
    dir=args.data_dir, split='val', block_size=args.block_size
)
owt_train_dl = DataLoader(owt_train_ds, batch_size=args.batch_size)
owt_val_dl = DataLoader(owt_val_ds, batch_size=args.batch_size)

# Init iteration number and best loss. Can override later if init_from = 'resume'
iter_num = 0
best_val_loss = 1e9

# Get vocab_size from meta file, or set to GPT2 default (50257)
meta_path = os.path.join(args.data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"vocab_size = {vocab_size} (from {meta_path})")
else:
    print(f"vocab_size not found in {meta_path}, using GPT-2 default of 50257")
    vocab_size = 50257

# Model init
model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                  block_size=args.block_size, dropout=args.dropout, 
                  vocab_size=vocab_size, bias=args.bias)
if args.init_from == 'scratch': # New model from scratch
    print('Initializing a new model from scratch')
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif args.init_from == 'resume':
    print(f'Resuming training from {args.out_dir}')
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Compare model args with checkpoint model args
    for k, v in model_args.items():
        assert checkpoint_model_args[k] == v, f'checkpoint model arg: {k} do not match model arg: {k}'
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Fix state dict keys, remove '_orig_mod.'
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif args.init_from.startswith('gpt2'):
    print(f'Initializing from OpenAI GPT2 weights: {args.init_from}')
    assert args.bias, 'GPT2 models have bias, set bias=True'
    override_args = dict(dropout=args.dropout)
    model = GPT.from_pretrained(args.init_from, override_args)
    model_args['n_layer'] = model.config.n_layer
    model_args['n_head'] = model.config.n_head
    model_args['n_embd'] = model.config.n_embd
# Crop block size
if args.block_size < model.config.block_size:
    model.crop_block_size(args.block_size)
model.to(device)

# Init GradScaler if data type is float16
scaler = None
if args.dtype == 'float16':
    print(f'Initializing Gradient Scaler to account for dtype: {args.dtype}')
    scaler = torch.cuda.amp.GradScaler()

# Optimizer
optimizer = model.configure_optimizers(
    args.weight_decay, args.learning_rate, (args.beta1, args.beta2)
)
if args.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Compile the model
if args.compile:
    print('Compiling the model...')
    unoptimized_model = model
    model =torch.compile(model) # Requires PyTorch 2.0

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Define a function for accurate loss estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    # Calculate mean training loss
    train_losses = torch.zeros(args.eval_iters)
    for k in range(args.eval_iters):
        X, Y = next(iter(owt_train_dl))
        with ctx:
            logits, loss = model(X, Y)
        train_losses[k] = loss.item()
    out['train'] = train_losses.mean()
    # Calculate mean validation loss
    val_losses = torch.zeros(args.eval_iters)
    for k in range(args.eval_iters):
        X, Y = next(iter(owt_val_dl))
        with ctx:
            logits, loss = model(X, Y)
        val_losses[k] = loss.item()
    out['val'] = val_losses.mean()
    model.train()
    return out

# Define a learning rate scheduler (cosine with warmup)
def get_lr(iter):
    # During warmup iterations: linear
    if iter < args.warmup_iters:
        return args.learning_rate * iter / args.warmup_iters
    # After decay iterations: minimum
    if iter > args.lr_decay_iters:
        return args.min_lr
    # Decay iterations
    decay_ratio = (iter - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# Logging
if args.wandb_log and master_process:
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )

# Training loop
t0 = time.time()
while True:

    # Get learning rate
    if args.decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = args.learning_rate

    # If reached eval iteration, evaluate loss on train/val and write checkpoints
    if iter_num % args.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Log entry
        if args.wandb_log:
            wandb.log({
                'iter': iter_num,
                'train/loss': losses['train'],
                'val/loss': losses['val'],
                'lr': lr
            })
        # If new best validation loss, save checkpoint
        if losses['val'] < best_val_loss or args.always_save_checkpoint:
            best_val_loss = losses['val']
            raw_model = model.module if ddp else model
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                print(f'Saving checkpoint to {args.out_dir}')
                torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
    if iter_num == 0 and args.eval_only:
        break

    # Forward and backward pass with optional gradient accumulation to simulate 
    # larger batch size and GradScaler if using float16
    for micro_step in range(args.gradient_accumulation_steps):

        X, Y = next(iter(owt_train_dl))
        # Only sync gradients on last micro step
        if ddp: 
            model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
        
        with ctx:
            logits, loss = model(X, Y)

        scaler.scale(loss).backward() if scaler else loss.backward()
    
    # Clip gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer) if scaler else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    # Step the optimizer
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # Flush gradients
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % args.log_interval == 0 and master_process:
        lossf = loss.item()
        print(f'iter {iter_num}: loss {lossf:.4f}, time{dt*1000:.2f}ms')
    iter_num +=1

    # Termination conditions
    if iter_num > args.max_iters:
        break

if ddp:
    destroy_process_group()
