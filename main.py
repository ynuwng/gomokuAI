# main.py

import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
from gomoku import Gomoku
from gomoku_ai import GomokuNN, GomokuAI

class GomokuGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku")
        self.board_size = 15
        self.cell_size = 40
        self.gomoku = Gomoku(self.board_size)

        # Load the trained model
        self.model = GomokuNN(self.board_size)
        self.model.load_state_dict(torch.load('gomoku_ai_model.pth'))
        self.model.eval()

        self.ai = GomokuAI(self.model, 2, self.board_size)  # AI is player 2

        self.canvas = tk.Canvas(self.root, width=self.board_size * self.cell_size, height=self.board_size * self.cell_size)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_board()
        self.update_board()

    def draw_board(self):
        for i in range(self.board_size):
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.board_size * self.cell_size)
            self.canvas.create_line(0, i * self.cell_size, self.board_size * self.cell_size, i * self.cell_size)

    def update_board(self):
        self.canvas.delete("piece")
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.gomoku.board[x][y] == 1:
                    self.canvas.create_oval(x * self.cell_size + 5, y * self.cell_size + 5,
                                            (x + 1) * self.cell_size - 5, (y + 1) * self.cell_size - 5,
                                            fill="black", tags="piece")
                elif self.gomoku.board[x][y] == 2:
                    self.canvas.create_oval(x * self.cell_size + 5, y * self.cell_size + 5,
                                            (x + 1) * self.cell_size - 5, (y + 1) * self.cell_size - 5,
                                            fill="white", tags="piece")

    def on_click(self, event):
        x, y = event.x // self.cell_size, event.y // self.cell_size
        if self.gomoku.is_valid_move(x, y):
            result = self.gomoku.make_move(x, y)
            self.update_board()
            if result != 'Continue':
                messagebox.showinfo("Game Over", result)
                self.gomoku.reset()
                self.update_board()
                return

            # AI makes a move after the player
            self.ai_move()

    def ai_move(self):
        ai_move = self.ai.get_best_move(self.gomoku)
        if ai_move is not None and self.gomoku.is_valid_move(ai_move[0], ai_move[1]):
            result = self.gomoku.make_move(ai_move[0], ai_move[1])
            self.update_board()
            if result != 'Continue':
                messagebox.showinfo("Game Over", result)
                self.gomoku.reset()
                self.update_board()

if __name__ == "__main__":
    root = tk.Tk()
    game = GomokuGame(root)
    root.mainloop()
