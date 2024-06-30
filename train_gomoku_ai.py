# train_gomoku_ai.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gomoku_ai import GomokuNN

def train_model(model, data, labels, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for board, label in zip(data, labels):
            board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
            label_tensor = torch.FloatTensor(label).unsqueeze(0)
            optimizer.zero_grad()
            output = model(board_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == "__main__":
    board_size = 15
    model = GomokuNN(board_size)
    model.load_state_dict(torch.load('gomoku_ai_model.pth'))
    model.eval()

    # Load self-play data
    data_file = np.load('self_play_data.npz')
    data = data_file['data']
    labels = data_file['labels']

    # Train the model with self-play data
    train_model(model, data, labels, epochs=50)

    # Save the improved model
    torch.save(model.state_dict(), 'gomoku_ai_model.pth')
