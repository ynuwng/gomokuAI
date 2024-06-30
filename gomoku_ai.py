# gomoku_ai.py

import torch
import torch.nn as nn
import numpy as np


class GomokuNN(nn.Module):
    def __init__(self, board_size):
        super(GomokuNN, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, board_size * board_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GomokuAI:
    def __init__(self, model, player, board_size):
        self.model = model
        self.player = player
        self.board_size = board_size

    def get_best_move(self, game):
        board = np.array(game.board)
        board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(board_tensor).view(self.board_size, self.board_size)

        # Sort moves by score in descending order
        moves = \
        np.dstack(np.unravel_index(np.argsort(-output.cpu().numpy().ravel()), (self.board_size, self.board_size)))[0]

        for move in moves:
            if game.is_valid_move(move[0], move[1]):
                return move

        return None
