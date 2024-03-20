from transformers import GPT2Tokenizer, GPT2ForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from dataset import get_train_tokens, get_train_labels
import torch
from torch.nn.utils.rnn import pad_sequence

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device.type)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2ForTokenClassification.from_pretrained('gpt2')
print("Model and tokenizer loaded")

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move the model to the GPU if available
model = model.to(device)

# The issue in your code is that the length of your input sequences might be exceeding the maximum sequence length that the GPT-2 model can handle (1024 tokens). To fix this, you need to ensure that your input sequences do not exceed this limit during the tokenization process.

# Convert your tokens to tensors and pad the sequences
train_tokens = [torch.tensor(tokenizer.encode(tokens, truncation=True, padding='max_length', max_length=model.config.max_position_embeddings)) for tokens in get_train_tokens()]

# Pad the sequences
train_tokens = pad_sequence(train_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)

# Your padded sequences are your input_ids
input_ids = train_tokens

# Create attention_mask
attention_mask = input_ids != tokenizer.pad_token_id

# Convert your labels to tensors and pad them to match the length of your input sequences
labels = [torch.tensor(label[:model.config.max_position_embeddings] + [-100]*(model.config.max_position_embeddings-len(label))) for label in get_train_labels()]

# Pad the labels
labels = pad_sequence(labels, batch_first=True, padding_value=-100)

# Create a TensorDataset from the inputs and labels
dataset = TensorDataset(input_ids, attention_mask, labels)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=16)

# Fine-tune the model
model.train()
optim = AdamW(model.parameters(), lr=5e-5)

# Define the number of epochs
epochs = 3

# Training loop
for epoch in range(epochs):
    for batch in data_loader:
        print(f"Running batch: {batch}")
        # Move the batch tensors to the same device as the model
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        print(input_ids.shape)
        print(attention_mask.shape)
        print(labels.shape)

        print(f"Max position embeddings: {model.config.max_position_embeddings}")
        print(f"Max input sequence length: {input_ids.shape[1]}")

        exit()

        optim.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
    print(f'Epoch {epoch + 1}/{epochs} done')

# Save the model
model.save_pretrained('gpt2-trained-ner-model')