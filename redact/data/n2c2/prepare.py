import os
import json
from bs4 import BeautifulSoup as bs
import re
import numpy as np
import tiktoken
from tqdm import tqdm


# I/O
data_dir = '/media/md0-raid1/n2c2/'
out_train_path = '/media/nvme2/n2c2/train.npy'
out_val_path = '/media/nvme2/n2c2/val.npy'
with open('label2id.json') as f:
    label2id = json.load(f)

## 2014 Dataset
print('Preparing n2c2 2014 deidentification dataset')

# Define function for encoding one record
def encode_single_record14(record):
    '''
    Encodes one medical record from the n2c2 2014 deidentification dataset.
    Currently only supports byte pair encoding used by GPT2.

    Returns two lists: (1) The token ids of the record text and (2) the labels cooresponding to each token.
    '''
    # Get text and labels from record object
    text = record.find('text')
    text_w_labels = str(text)
    text_wout_labels = str(text.text)
    phis = text.find_all('phi')

    # Get start and end locations
    start_idx = 0
    end_idx = len(text_w_labels)

    # Init empty array for phi labels
    phi_array = []
    for phi in phis: # Iterate through phi labels
        phi_type = phi['type'] # Get label type
        phi_text = phi.text # Get label text
        for phi_text_token in phi_text.split(): # Split on spaces and iterate through tokens
            phi_text_len = len(phi_text_token) # Get length of each token
            phi_start = str(text).find(phi_text_token, start_idx, end_idx) # Find start position of phi token in record text
            phi_end = phi_start + phi_text_len # Find end postion of phi token in record text
            # Here, we make adjustments to DATE-type phi labels to include the trailing year
            if (phi_type == 'DATE') & bool(re.match(r'(0[1-9]|1[0-2])\/(0[1-9]|1\d|2\d|3[01])', phi_text_token)):
                if bool(re.match(r'\/\d\d\d\d', str(text)[phi_end+6: phi_end+11])):
                    phi_text_token = phi_text_token + str(text)[phi_end+6: phi_end+11]
                    phi_text_len += 5
                    phi_end += 11
                elif bool(re.match(r'\/\d\d', str(text)[phi_end+6: phi_end+9])):
                    phi_text_token = phi_text_token + str(text)[phi_end+6: phi_end+9]
                    phi_text_len += 3
                    phi_end += 9
            start_idx = phi_end # Update start_idx so we progress through record
            phi_array.append([phi_type, phi_text_token]) # Add label and text of phi token to phi label array
        
    ids_out = np.array([], dtype=np.uint16) # Init empty list to store BPE encoded tokens
    labels_out = np.array([], dtype=np.uint8) # Init empty list to store token-level labels
    phi_idx = 0 # For iterating through phi_array
    for token in text_wout_labels.split(): # Iterate through each word (split by spaces)
        token_ids = np.array(enc.encode(token + ' '), dtype=np.uint16) # BPE encode the token and trailing space
        ids_out = np.append(ids_out, token_ids)
        if phi_idx == len(phi_array): # If we've reached the last phi for this record, append 'O' labels to labels_out
            labels_out = np.append(labels_out, np.array([label2id['O']] * len(token_ids), dtype=np.uint8))
        elif token == phi_array[phi_idx][1]: # If the current token matches the next phi test, append appropriate labels (eg. ['B-ID', 'I-ID', 'O']) to labels_out
            begin_label = 'B-' + phi_array[phi_idx][0]
            begin_label_id = label2id[begin_label]
            inner_label = 'I-' + phi_array[phi_idx][0]
            inner_label_id = label2id[inner_label]
            # Note: The first label starts with 'B-' and inner labels start with 'I-'
            # Note: Since the final id will always represent a space, we end each list of labels with an 'O'
            new_labels = [begin_label_id] + [inner_label_id] * (len(token_ids) - 2) + [label2id['O']]
            labels_out = np.append(labels_out, np.array(new_labels, dtype=np.uint8))
            phi_idx += 1 # Since we matched a phi token, move to next
        else:
            labels_out = np.append(labels_out, np.array([label2id['O']] * len(token_ids), dtype=np.uint8))

    # Make sure ids and labels have same number of elements
    assert len(ids_out) == len(labels_out), 'Length of output ids must match length of output labels'

    return ids_out, labels_out

# Read and parse xml file
data_path = os.path.join(
    data_dir,
    'n2c2_train/deid_surrogate_train_all_version2.xml'
)
with open(data_path, 'r') as file:
    content = file.readlines()
content = ''.join(content)
bs_content = bs(content, 'lxml')
records = bs_content.find_all('record')

# Init encoder
enc = tiktoken.get_encoding('gpt2')

# Init id and label arrays
train_ids = np.array([], dtype=np.uint16)
train_labels = np.array([], dtype=np.uint16)
val_ids = np.array([], dtype=np.uint16)
val_labels = np.array([], dtype=np.uint16)

# Split records into train and val
shuffle_ids = np.random.RandomState(seed=77).permutation(len(records))
split_id = int(len(records) * 0.8)
train_record_ids = shuffle_ids[:split_id]
val_record_ids = shuffle_ids[split_id:]
print(f'''
Total records in 2014 train: {len(train_record_ids)}
Total records in 2014 val: {len(val_record_ids)}
''')

# Encode each record and append to arrays
for split in ['train', 'val']:
    record_ids = train_record_ids if split == 'train' else val_record_ids
    for record_id in tqdm(record_ids):
        record = records[record_id]
        ids, labels = encode_single_record14(record)
        if split == 'train':
            train_ids = np.append(train_ids, ids)
            train_labels = np.append(train_labels, labels)
        else:
            val_ids = np.append(val_ids, ids)
            val_labels = np.append(val_labels, labels)

train14_size = len(train_ids)
val14_size = len(val_ids)
print(f'''
Total ids in  2014 train: {train14_size},
Total ids in 2014 val: {train14_size}
''')

## 2006
print('Preparing n2c2 2006 deidentification dataset')

# Define function for encoding one record
def encode_single_record06(record):
    '''
    '''
    # Get text and labels from record object
    text = record.find('text').text[9:-3]
    tags = record.find('tags')
    phis = [
        [phi['type'], phi['text'], int(phi['start']), int(phi['end'])] for phi in tags if phi != '\n'
    ]
    # Init empty arrays for token ids and label ids
    ids_out = np.array([], dtype=np.uint16)
    labels_out = np.array([], dtype=np.uint8)
    start_idx = 0
    for phi in phis:
        # Encode text between previous phi/start index and current phi
        leading_text = text[start_idx:phi[2]]
        leading_text_ids = np.array(enc.encode(leading_text), dtype=np.uint16)
        # Set labels for leading text
        leading_label_ids = np.array([label2id['O']] * len(leading_text_ids), dtype=np.uint8)
        # Append leading text ids and labels
        ids_out = np.append(ids_out, leading_text_ids)
        labels_out = np.append(labels_out, leading_label_ids)

        # Encode phi text
        phi_text = text[phi[2]: phi[3]]
        if phi_text != phi[1]: # Make sure PHI text matches actual text in that location
             return None, None, True
        phi_text_ids = np.array(enc.encode(phi_text), dtype=np.uint16)
        # Set labels for phi text
        begin_label = 'B-' + phi[0]
        begin_label_id = label2id[begin_label]
        inner_label = 'I-' + phi[0]
        inner_label_id = label2id[inner_label]
        phi_label_ids = np.array(
            [begin_label_id] + [inner_label_id] * (len(phi_text_ids) - 1),
            dtype=np.uint8
        )
        # Append phi text ids and labels
        ids_out = np.append(ids_out, phi_text_ids)
        labels_out = np.append(labels_out, phi_label_ids)

        # Update start_idx
        start_idx = phi[3]

    # Encode and label any text remaining after last phi tag
    remaining_text = text[start_idx:]
    remaining_text_ids = np.array(enc.encode(remaining_text), dtype=np.uint16)
    # Set labels for remaining text
    remaining_label_ids = np.array([label2id['O']] * len(remaining_text_ids), dtype=np.uint8)
    # Append remaining text ids and labels
    ids_out = np.append(ids_out, remaining_text_ids)
    labels_out = np.append(labels_out, remaining_label_ids)
    if len(ids_out) != len(labels_out): # Make sure length of ids matches length of labels
        return None, None, True

    return ids_out, labels_out, False

# Read and parse xml file
data_dir = os.path.join(
    data_dir,
    'training-PHI-Gold-Set1'
)

# Get list of record ids
record_fnames = [fname for fname in os.listdir(data_dir)]

# Init encoder
enc = tiktoken.get_encoding('gpt2')

# Split records into train and val
shuffle_ids = np.random.RandomState(seed=77).permutation(len(record_fnames))
split_id = int(len(record_fnames) * 0.8)
train_record_ids = shuffle_ids[:split_id]
val_record_ids = shuffle_ids[split_id:]
print(f'''
Total records in 2006 train: {len(train_record_ids)}
Total records in 2006 val: {len(val_record_ids)}
''')

broken_ids = []
# Encode each record and append to arrays
for split in ['train', 'val']:
    record_ids = train_record_ids if split == 'train' else val_record_ids
    for record_id in tqdm(record_ids): # Iterate through records
        # Get record
        record_fname = record_fnames[record_id]
        record_path = os.path.join(data_dir, record_fname)
        with open(os.path.join(data_dir, f'{record_fname}'), 'r') as f:
            content = f.readlines()
        content = ''.join(content)
        record = bs(content, 'lxml')
        # Encode record
        ids, labels, record_broken = encode_single_record06(record)
        if record_broken:  # If encoding fails, append record name to broken_id list
            broken_ids.append(record_fname)
        else: # Append token ids and label ids to output arrays
            if split == 'train':
                train_ids = np.append(train_ids, ids)
                train_labels = np.append(train_labels, labels)
            else:
                val_ids = np.append(val_ids, ids)
                val_labels = np.append(val_labels, labels)

train06_size = len(train_ids) - train14_size
val06_size = len(val_ids) - val14_size
print(f'''
Total ids in  2006 train: {train06_size},
Total ids in 2006 val: {val06_size}
''')

# Stack ids and labels together
train_np = np.stack((train_ids, train_labels), axis=-1, dtype=np.uint16)
val_np = np.stack((val_ids, val_labels), axis=-1, dtype=np.uint16)
print(f'''
Shape of final train: {train_np.shape},
Shape of final val: {val_np.shape}
''')

# Write final output files
print(f'Saving output files to:\n\t{out_train_path}\n\t{out_val_path}')
np.save(out_train_path, train_np)
np.save(out_val_path, val_np)

print(f"Unable to process the following files:\n{', '.join(broken_ids)}")