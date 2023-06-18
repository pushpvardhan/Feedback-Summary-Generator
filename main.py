import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sympy.physics.units import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.throughput_benchmark import format_time
from transformers import AdamW
from transformers import AutoModel, BertTokenizerFast

from BERT_Arch import BERT_Arch

device = torch.device("cuda")
df = pd.read_excel("Peer Form 1.xlsx")
# print(df.to_string())
df1 = df.drop(['Options'], axis=1)
# print(df1.to_string())


# df1['Criteria'] = df1['Criteria'].astype('category')
# df1['Select Option'] = df1['Select Option'].astype('category')
# df1['Cite Examples and Additional Comments - As Applicable.'] = df1['Cite Examples and Additional Comments - As Applicable.'].astype('category')
df1['Overall Summary'] = df1['Overall Summary'].astype('category')

# df1['Criteria'] = df1['Criteria'].cat.codes
# df1['Select Option'] = df1['Select Option'].cat.codes
# df1['Cite Examples and Additional Comments - As Applicable.'] = df1['Cite Examples and Additional Comments - As Applicable.'].cat.codes
df1['Overall Summary'] = df1['Overall Summary'].cat.codes
# print(df1.to_string())

# print(df1.iloc)
print("**************************")
# print(df1.iloc[0])

train_X, temp_X, train_summary, temp_summary = train_test_split(df1.iloc[:,[3, 4, 5]], df1.iloc[:,6], random_state=0, train_size=0.5, stratify=df1.iloc[:,6])
print(train_X)
val_X, test_X, val_summary, test_summary = train_test_split(temp_X, temp_summary, random_state=0,
    test_size=0.5)
    # stratify=temp_summary.iloc[:,0])

bert = AutoModel.from_pretrained('bert-base-uncased')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

tokens_train_criteria = tokenizer.batch_encode_plus(train_X['Criteria'].to_list(), truncation=True, padding=True)
tokens_train_option = tokenizer.batch_encode_plus(train_X['Select Option'].to_list(), truncation=True, padding=True)
tokens_train_example = tokenizer.batch_encode_plus(train_X['Cite Examples and Additional Comments - As Applicable.'].to_list(), truncation=True, padding=True)
# print(tokens_train)
tokens_test_criteria = tokenizer.batch_encode_plus(train_X['Criteria'].to_list(), truncation=True, padding=True)
tokens_test_option = tokenizer.batch_encode_plus(train_X['Select Option'].to_list(), truncation=True, padding=True)
tokens_test_example = tokenizer.batch_encode_plus(train_X['Cite Examples and Additional Comments - As Applicable.'].to_list(), truncation=True, padding=True)
# tokens_test = tokenizer.batch_encode_plus(train_X.to_list(), truncation=True, padding=True)
tokens_val_criteria = tokenizer.batch_encode_plus(train_X['Criteria'].to_list(), truncation=True, padding=True)
tokens_val_option = tokenizer.batch_encode_plus(train_X['Select Option'].to_list(), truncation=True, padding=True)
tokens_val_example = tokenizer.batch_encode_plus(train_X['Cite Examples and Additional Comments - As Applicable.'].to_list(), truncation=True, padding=True)
# tokens_val = tokenizer.batch_encode_plus(train_X.to_list(), truncation=True, padding=True)
print('****************************************')
print(tokens_train_criteria)
print('****************************************')
print(tokens_train_option)
print('****************************************')
print(tokens_train_example)
print('****************************************')


train_seq_criteria = torch.tensor(tokens_train_criteria['input_ids'])
train_seq_option = torch.tensor(tokens_train_option['input_ids'])
train_seq_example = torch.tensor(tokens_train_example['input_ids'])
train_seq = torch.cat((train_seq_criteria, train_seq_option, train_seq_example), 1)
print(train_seq)
# train_seq = torch.cat(train_seq, out=)
train_mask_criteria = torch.tensor(tokens_train_criteria['attention_mask'])
train_mask_option = torch.tensor(tokens_train_option['attention_mask'])
train_mask_example = torch.tensor(tokens_train_example['attention_mask'])
train_mask = torch.cat((train_mask_criteria, train_mask_option, train_mask_example), 1)
train_y = torch.tensor(train_summary.tolist())

val_seq_criteria = torch.tensor(tokens_val_criteria['input_ids'])
val_seq_option = torch.tensor(tokens_val_option['input_ids'])
val_seq_example = torch.tensor(tokens_val_example['input_ids'])
val_seq = torch.cat((val_seq_criteria, val_seq_option, val_seq_example), 1)
val_mask_criteria = torch.tensor(tokens_val_criteria['attention_mask'])
val_mask_option = torch.tensor(tokens_val_option['attention_mask'])
val_mask_example = torch.tensor(tokens_val_example['attention_mask'])
val_mask = torch.cat((val_mask_criteria, val_mask_option, val_mask_example), 1)
val_y = torch.tensor(val_summary.tolist())

test_seq_criteria = torch.tensor(tokens_test_criteria['input_ids'])
test_seq_option = torch.tensor(tokens_test_option['input_ids'])
test_seq_example = torch.tensor(tokens_test_example['input_ids'])
test_seq = torch.cat((test_seq_criteria, test_seq_option, test_seq_example), 1)
test_mask_criteria = torch.tensor(tokens_test_criteria['attention_mask'])
test_mask_option = torch.tensor(tokens_test_option['attention_mask'])
test_mask_example = torch.tensor(tokens_test_example['attention_mask'])
test_mask = torch.cat((test_mask_criteria, test_mask_option, test_mask_example), 1)
test_y = torch.tensor(test_summary.tolist())

batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

for param in bert.parameters():
    param.requires_grad = False

model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

class_weights = compute_class_weight('balanced', classes=np.unique(train_summary), y=train_summary)
print("Class Weights:", class_weights)

weights = torch.tensor(class_weights, dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy = nn.NLLLoss(weight=weights)

# number of training epochs
epochs = 10


def train():
    model.train()
    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


def evaluate(t0=None):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses = []
valid_losses = []

# for each epoch
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # train model
    train_loss, _ = train()

    # evaluate model
    valid_loss, _ = evaluate()

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
    # model's performance
    preds = np.argmax(preds, axis=1)
    print(classification_report(test_y, preds))
