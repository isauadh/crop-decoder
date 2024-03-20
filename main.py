from transformers import GPT2Tokenizer, GPT2ForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from dataset import get_train_tokens, get_train_labels
import torch

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

# Get your training tokens & labels
train_tokens = get_train_tokens()
print(train_tokens[0])

train_labels = get_train_labels()
print(train_labels[0])

# Preprocess your tokens
inputs = tokenizer(train_tokens, truncation=True, padding=True, is_split_into_words=True, return_tensors='pt')

# Convert your labels to tensors
from torch.nn.utils.rnn import pad_sequence
labels = pad_sequence([torch.tensor(labels) for labels in train_labels], batch_first=True, padding_value=-100)

# Create a TensorDataset from the inputs and labels
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

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

        optim.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
    print(f'Epoch {epoch + 1}/{epochs} done')

# Save the model
model.save_pretrained('gpt2-trained-ner-model')