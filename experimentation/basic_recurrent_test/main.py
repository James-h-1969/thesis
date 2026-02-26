import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from models.SentimentRNN import SentimentRNN

if __name__ == "__main__":
    df = pd.read_csv("/data/IMDB Dataset.csv", names=["text", "label"])
    df['text'] = df['text'].str.lower().str.split()

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    vocab = {word for phrase in df['text'] for word in phrase}
    word_to_idx = {word: idx for idx, word in enumerate(vocab, start=1)}

    max_length = df['text'].str.len().max()

    def encode_and_pad(text):
        encoded = [word_to_idx[word] for word in text]
        return encoded + [0] * (max_length - len(encoded))

    train_data['text'] = train_data['text'].apply(encode_and_pad)
    test_data['text'] = test_data['text'].apply(encode_and_pad)

    class SentimentDataset(Dataset):
        def __init__(self, data):
            self.texts = data['text'].values
            self.labels = data['label'].values
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    train_dataset = SentimentDataset(train_data)
    test_dataset = SentimentDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    vocab_size = len(vocab) + 1
    embed_size = 128
    hidden_size = 128
    output_size = 2 
    model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for texts, labels in train_loader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')