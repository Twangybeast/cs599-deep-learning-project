import sys

import numpy as np
import tqdm

import chess
import chess.pgn

PIECE_TO_IDX = {p: i for i, p in enumerate(" PNBRQKpnbrqk")}

def board_to_npy(board):
    result = np.zeros((8, 8), dtype=int)
    piece_map = board.piece_map()
    for square, piece in board.piece_map().items():
        result[chess.square_rank(square)][chess.square_file(square)] = PIECE_TO_IDX[piece.symbol()]
    return result

def score_to_float(score):
    res = score.relative.score(mate_score=100000) / 100
    return max(min(res, 10), -10)

def get_game_info(game):
    boards = []
    scores = []
    for node in game.mainline():
        boards.append(board_to_npy(node.board()))
        scores.append(score_to_float(node.eval()))
    return np.array(boards), np.array(scores), np.array((float(game.headers['WhiteElo']) + float(game.headers['BlackElo'])) / 2)

def main():
    games = []
    elos = []
    time_controls = {}
    events = set()

    with tqdm.tqdm() as pbar:
        game = None
        count = 0
        eval_count = 0
        while True:
            game = chess.pgn.read_game(sys.stdin)
            if game is None:
                break
            pbar.update(1)
            count += 1
            # Filters
            if any((node.eval() is None for node in game.mainline())):
                continue
            if game.headers["TimeControl"] != "300+0":
                continue

            
            elos.append([float(game.headers['WhiteElo']), float(game.headers['BlackElo'])])
            for node in game.mainline():
                print(board_to_npy(node.board()))
            break


            # Process game
            
            # if "Rated Standard" not in game.headers['Event']:
            #     continue
            # games.append(game)
            # events.add(game.headers['Event'])
            # tc = game.headers['TimeControl']
            # time_controls[tc] = time_controls.get(tc, 0) + 1
            # elos.append([float(game.headers['WhiteElo']), float(game.headers['BlackElo'])])
            if count >= 10000:
                break
            # print(game)
    # print(games)
    # print(len(elos))
    # print(eval_count, count)
    # print(len(time_controls))
    # print(sorted([(t, f) for t, f in time_controls.items()], key=lambda x: -x[1]))
    # print(len(events))
    # print(sorted(list(events)))
    # np.save("elos_all.npy", elos)

if __name__ == '__main__':
    main()
