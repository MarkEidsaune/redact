# '''
# GPT2 model finetuning on n2c2 dataset.
# '''

import argparse
import os
import time
from datetime import datetime
import math
import pickle
import json
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from gpt2_model import GPTConfig, GPT, LayerNorm

dttm = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description=__doc__)
# I/O
parser.add_argument('--out-dir', default='/media/nvme2/models/redact', type=str, help='Output directory for saving checkpoints.')
parser.add_argument('--model-fname', default='ckpt_owt.pt', type=str, help='Filename of model checkpoint.')
parser.add_argument('--labels-dir', default='data/n2c2/label2id.json', type=str, help='Path to n2c2 labels to label ids map')
parser.add_argument('--eval-interval', default=1000, type=int, help='Number of iterations between evaluations.')
parser.add_argument('--log-interval', default=100, type=int, help='Number of iterations between log entries.')
parser.add_argument('--eval-iters', default=200, type=int, help='Number of iterations to perform for each evaluation.')
parser.add_argument('--eval-only', action='store_true', help='Default = false. If true, script exits after the first evaluation.')
parser.add_argument('--always-save-checkpoint', action='store_true', help='Default = false. If true, save checkpoint after each evaluation.  Otherwise, save only after beating best validation loss.')
# Logging
parser.add_argument( '--wandb-log', action='store_true', help='Default = false.  If true, log results using wandb account.')
parser.add_argument('--wandb-project', default='gpt2_finetuning_n2c2', type=str, help='Wandb project name.')
parser.add_argument('--wandb-run-name', default=dttm, type=str, help='Wandb run name.')
# Data
parser.add_argument('--dataset', default='n2c2', type=str, help='Name of training dataset.')
parser.add_argument('--data-dir', default='/media/nvme2/n2c2', type=str, help='Data directory for training dataset.')
parser.add_argument('--gradient-accumulation-steps', default=4, type=int, help='Used to simulate larger batch sizes.')
parser.add_argument('--batch-size', default=3, type=int, help='Batch size.  If gradient-accumulation-steps > 1, this is the micro-batch size.')
parser.add_argument('--block-size', default=1024, type=int, help='Number of input tokens.')
# Model
parser.add_argument('--n-layer', default=12, type=int, help='Number of transformer blocks in model.')
parser.add_argument('--n-head', default=12, type=int, help='Number of attention heads in each transformer block.')
parser.add_argument('--n-embd', default=768, type=int, help='Length of embedding vectors (token & position).')
parser.add_argument('--dropout', default=0.0, type=float, help='For pretraining, 0 is good, for finetuning try 0.1+.')
parser.add_argument('--bias', action='store_true', help='Default = false.  If true, use bias in LayerNorm and Linear layers.')
parser.add_argument('--freeze', action='store_true', help='Default = false.  If true, freeze all parameters except those of the last 3 linear layers of the head.')
parser.add_argument('--replace-head', action='store_true', help='Default = false.  If true, replace head with original head + 3 new linear layers that reduce output dimensionality to n_labels.')
# Optimizer
parser.add_argument('--learning-rate', default=6e-5, type=float, help='Maximum learning rate.')
parser.add_argument('--max-iters', default=100000, type=int, help='Total number of training iterations.')
parser.add_argument('--weight-decay', default=1e-2, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.95, type=float)
parser.add_argument('--grad-clip', default=1.0, type=float, help='Clip gradients at this value, or disable if = 0.0.')
# Learning rate decay
parser.add_argument('--decay-lr', action='store_false', help='Default = true. If false, do not decay learning rate.')
parser.add_argument('--warmup-iters', default=1000, type=int)
parser.add_argument('--lr-decay-iters', default=100000, type=int, help='Should be ~ max-iters.')
parser.add_argument('--min-lr', default=6e-6, type=float, help='Should be ~ learning_rate/10.')
# DDP settings
parser.add_argument('--backend', default='nccl', type=str, help='\'nccl\', \'gloo\', etc.')
parser.add_argument('--device', default='cuda', type=str, help='\'cpu\', \'cuda\', \'cuda:0\', \'cuda:1\', etc.')
parser.add_argument('--dtype', default='float16', type=str, help='\'float32\', \'bfloat16\', or \'float16\'. The latter will auto-implement a GradScaler.')
parser.add_argument('--compile', action='store_false', help='Default = true.  If false, do not use PyTorch 2.0 to compile the model to be faster.')

args = parser.parse_args()
config = vars(args)
print(config)

# Set up distributed training
ddp = int(os.environ.get('RANK', -1)) != -1 # Is this a ddp run?
if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # This process will do logging, checkpointing etc.
    seed_offset = ddp_rank # Each process gets a different seed
else:
    # Single gpu, single process
    master_process = True
    device = args.device
    seed_offset = 0

if master_process:
    os.makedirs(args.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # Allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # Allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # For later us in torch.autocast
# Note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Simple data loader
train_data = np.load(os.path.join(args.data_dir, 'train.npy'))
val_data = np.load(os.path.join(args.data_dir, 'val.npy'))
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.block_size, 0]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i:i+args.block_size, 1]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Init iteration number and best loss. Can override later if init_from = 'resume'
iter_num = 0
best_val_loss = 1e9

# Get number of labels
with open(args.labels_dir, 'r') as f:
    label2id = json.load(f)
n_labels = max(label2id.values())

# Model init
model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                  block_size=args.block_size, dropout=args.dropout, 
                  vocab_size=None, bias=args.bias)
ckpt_path = os.path.join(args.out_dir, args.model_fname)
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# Compare model args with checkpoint model args
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
# Fix state dict keys, remove '_orig_mod.'
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
# Replace head
if args.replace_head:
    print('Replacing model head')
    model.replace_head(n_labels)
# Crop block size
if args.block_size < model.config.block_size:
    model.crop_block_size(args.block_size)
    model_args['block_size'] = args.block_size
# Freeze all parameters except new layers of head
if args.freeze:
    print('Freezing pretrained layers')
    model.freeze_layers()

model.to(device)

# Init GradScaler if data type is float16
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(
    args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
)

# Compile the model
if args.compile:
    print('Compiling the model')
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
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Define a learning rate scheduler (cosine with warmup)
def get_lr(it):
    # During warmup iterations: linear
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # After decay iterations: minimum
    if it > args.lr_decay_iters:
        return args.min_lr
    # Decay iterations
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
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
X, Y = get_batch('train') # Get first batch
t0 = time.time()

while True:

    # Get learning rate
    lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
                torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt_n2c2_frozen.pt'))

    if iter_num == 0 and args.eval_only:
        break

    # Forward and backward pass with optional gradient accumulation to simulate 
    # larger batch size and GradScaler if using float16
    for micro_step in range(args.gradient_accumulation_steps):

        # Only sync gradients on last micro step
        if ddp: 
            model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
        
        with ctx:
            _, loss = model(X, Y)

        # Async prefetch next batch while model is doing forward pass
        X, Y = get_batch('train')

        scaler.scale(loss).backward()
    
    # Clip gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    # Step the optimizer
    scaler.step(optimizer)
    scaler.update()

    # Flush gradients
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % args.log_interval == 0 and master_process:
        lossf = loss.item()
        print(f'iter {iter_num}: loss {lossf:.4f}, time{dt*1000:.2f}ms')

    iter_num += 1

    # Termination conditions
    if iter_num > args.max_iters:
        break

if ddp:
    destroy_process_group()
