import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuNN(nn.Module):
    def __init__(self, board_size):
        super(GomokuNN, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, board_size * board_size)        # Second fully connected layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path, board_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GomokuNN(board_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
def get_best_move(self, game):
    # Convert game board to a tensor suitable for the model
    board_tensor = torch.tensor(game.board, dtype=torch.float32)
    board_tensor = board_tensor.unsqueeze(0).unsqueeze(0).to(self.model.device)  # Add batch and channel dimensions

    with torch.no_grad():
        predictions = self.model(board_tensor).view(self.board_size, self.board_size)
    predictions = predictions.cpu().numpy()  # Convert predictions to numpy array for easier manipulation

    # Select the best valid move
    best_move = None
    max_score = float('-inf')
    for x in range(self.board_size):
        for y in range(self.board_size):
            if game.is_valid_move(x, y) and predictions[x][y] > max_score:
                max_score = predictions[x][y]
                best_move = (x, y)

    return best_move
class GomokuAI:
    def __init__(self, model_path, board_size):
        self.board_size = board_size
        self.model = load_model(model_path, board_size)
        self.model.device = next(self.model.parameters()).device  # Get model device

    def get_best_move(self, game):
        return self.get_best_move(game)
