import pygame
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pygame import gfxdraw


# AI model classes and functions
class GomokuNN(nn.Module):
    def __init__(self, board_size):
        super(GomokuNN, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, board_size * board_size)

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


class GomokuAI:
    def __init__(self, model_path, board_size):
        self.board_size = board_size
        self.model = load_model(model_path, board_size)
        self.model.device = next(self.model.parameters()).device

    def get_best_move(self, game):
        board_tensor = torch.tensor(game.board, dtype=torch.float32)
        board_tensor = board_tensor.unsqueeze(0).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            predictions = self.model(board_tensor).view(self.board_size, self.board_size)
        predictions = predictions.cpu().numpy()

        best_move = None
        max_score = float('-inf')
        for x in range(self.board_size):
            for y in range(self.board_size):
                if game.is_valid_move(x, y) and predictions[x][y] > max_score:
                    max_score = predictions[x][y]
                    best_move = (x, y)

        return best_move


# Gomoku game class
class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = [[0] * board_size for _ in range(board_size)]
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def is_valid_move(self, x, y):
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == 0

    def make_move(self, x, y):
        if self.is_valid_move(x, y):
            self.board[x][y] = self.current_player
            if self.check_win(x, y):
                self.game_over = True
                self.winner = self.current_player
            else:
                self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)
            return True
        return False

    def check_win(self, x, y):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][
                    ny] == self.current_player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][
                    ny] == self.current_player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False


# GUI class
class GomokuGUI:
    def __init__(self, game, ai):
        self.game = game
        self.ai = ai
        self.cell_size = 40
        self.width = self.cell_size * self.game.board_size
        self.height = self.cell_size * self.game.board_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku")

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BOARD_COLOR = (200, 200, 200)  # Light grey

    def draw_board(self):
        self.screen.fill(self.BOARD_COLOR)
        for i in range(self.game.board_size):
            pygame.draw.line(self.screen, self.BLACK, (i * self.cell_size, 0), (i * self.cell_size, self.height))
            pygame.draw.line(self.screen, self.BLACK, (0, i * self.cell_size), (self.width, i * self.cell_size))

        for x in range(self.game.board_size):
            for y in range(self.game.board_size):
                center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
                if self.game.board[x][y] == 1:
                    gfxdraw.aacircle(self.screen, *center, self.cell_size // 2 - 2, self.BLACK)
                    gfxdraw.filled_circle(self.screen, *center, self.cell_size // 2 - 2, self.BLACK)
                elif self.game.board[x][y] == 2:
                    gfxdraw.aacircle(self.screen, *center, self.cell_size // 2 - 2, self.WHITE)
                    gfxdraw.filled_circle(self.screen, *center, self.cell_size // 2 - 2, self.WHITE)

    def show_winner_message(self, winner):
        font = pygame.font.Font(None, 36)
        if winner == 1:
            text = font.render("You win!", True, self.BLACK, self.WHITE)
        else:
            text = font.render("AI wins!", True, self.BLACK, self.WHITE)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(3000)  # Display the message for 3 seconds

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game.game_over:
                    x, y = event.pos
                    board_x, board_y = x // self.cell_size, y // self.cell_size
                    if self.game.make_move(board_x, board_y):
                        if self.game.game_over:
                            self.draw_board()
                            self.show_winner_message(self.game.winner)
                        else:
                            ai_move = self.ai.get_best_move(self.game)
                            if ai_move:
                                self.game.make_move(*ai_move)
                                if self.game.game_over:
                                    self.draw_board()
                                    self.show_winner_message(self.game.winner)

            self.draw_board()
            pygame.display.flip()


# Main game loop
if __name__ == "__main__":
    board_size = 15
    model_path = "gomoku_ai_model.pth"  # Replace with the path to your trained model
    game = Gomoku(board_size)
    ai = GomokuAI(model_path, board_size)
    gui = GomokuGUI(game, ai)
    gui.run()