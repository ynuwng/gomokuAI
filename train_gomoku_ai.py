# train_gomoku_ai.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gomoku_ai import GomokuNN


def train_model(model, data, labels, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for board, label in zip(data, labels):
            board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0).to(device)
            label_tensor = torch.FloatTensor(label).unsqueeze(0).to(device)

            optimizer.zero_grad()
            output = model(board_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data)}')


if __name__ == "__main__":
    board_size = 15
    model = GomokuNN(board_size)

    # Load the trained model if available
    try:
        model.load_state_dict(torch.load('gomoku_ai_model.pth'))
    except FileNotFoundError:
        print("No existing model found, starting training from scratch.")

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load self-play data
    data_file = np.load('self_play_data.npz')
    data = data_file['data']
    labels = data_file['labels']

    # Train the model with self-play data
    train_model(model, data, labels, epochs=50, device=device)

    # Save the improved model
    torch.save(model.state_dict(), 'gomoku_ai_model.pth')
