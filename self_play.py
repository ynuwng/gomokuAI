# self_play.py

import numpy as np
import torch
from gomoku import Gomoku
from gomoku_ai import GomokuNN


def generate_self_play_data(model, num_games=100):
    data = []
    labels = []

    for _ in range(num_games):
        game = Gomoku()
        states = []
        moves = []

        while True:
            board = np.array(game.board)
            board_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                output = model(board_tensor).view(game.board_size, game.board_size)

            # Get valid moves
            valid_moves = [(x, y) for x in range(game.board_size) for y in range(game.board_size) if
                           game.is_valid_move(x, y)]
            move_probs = output.cpu().numpy().flatten()
            move_probs = np.array([move_probs[x * game.board_size + y] for x, y in valid_moves])

            # Ensure non-negative probabilities
            move_probs = np.maximum(move_probs, 0)

            # Normalize the probabilities
            if np.sum(move_probs) > 0:
                move_probs = move_probs / np.sum(move_probs)
            else:
                move_probs = np.ones(len(valid_moves)) / len(valid_moves)

            move = valid_moves[np.random.choice(len(valid_moves), p=move_probs)]

            states.append(board)
            moves.append(move)

            result = game.make_move(move[0], move[1])
            if result != 'Continue':
                winner = game.current_player if 'wins' in result else 0
                for state, move in zip(states, moves):
                    label = np.zeros((game.board_size, game.board_size))
                    label[move] = 1 if winner == 1 else -1
                    data.append(state)
                    labels.append(label.flatten())
                break

    return data, labels


if __name__ == "__main__":
    board_size = 15
    model = GomokuNN(board_size)
    model.load_state_dict(torch.load('gomoku_ai_model.pth'))
    model.eval()

    data, labels = generate_self_play_data(model, num_games=100)
    np.savez('self_play_data.npz', data=data, labels=labels)
