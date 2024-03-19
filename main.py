from transformers import GPT2Tokenizer, GPT2ForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from dataset import get_train_tokens, get_train_labels
import torch

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2ForTokenClassification.from_pretrained('gpt2')

# Move the model to the GPU if available
model = model.to(device)

# Get your training tokens & labels
train_tokens = get_train_tokens()
train_labels = get_train_labels()

# Preprocess your tokens
inputs = tokenizer(train_tokens, truncation=True, padding=True, is_split_into_words=True, return_tensors='pt')

# Convert your labels to tensors
labels = torch.tensor(train_labels)

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
        # Move the batch tensors to the same device as the model
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optim.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

# Save the model
model.save_pretrained('gpt2-trained-ner-model')