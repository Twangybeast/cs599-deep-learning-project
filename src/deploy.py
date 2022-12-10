import argparse
import io

import chess
import chess.pgn
import chess.engine
from flask import flask, request
import torch

from main import ChessNet
from prepare_games import get_game_info

# lol run this in the src folder
FILTER_SIZE = 256
NUM_LAYERS = 10
MAX_GAME_LENGTH = 12
MODEL_PATH = "...?"

state_dict = torch.load(MODEL_PATH)
model = ChessNet(FILTER_SIZE, NUM_LAYERS, MAX_GAME_LENGTH)
model.load_state_dict(state_dict['model'])
model.eval()

engine = chess.engine.SimpleEngine.popen_uci("TODO Stockfish path")


app = Flask(__name__)

@app.route("/game-elo")
def get_elo():
    try:
        pgn = request.args.get("pgn", default="", type=str)
        pgn = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn)
        game.headers['WhiteElo'] = "0"
        game.headers['BlackElo'] = "0"
        for node in game.mainline():
            info = engine.analyze(node.board(), chess.engine.Limit(time=0.2))
            node.set_eval(info['score'])
        boards, scores, elo = get_game_info(game)
        plane_tensor = game_to_tensor(boards, max(0, len(boards) - MAX_GAME_LENGTH), MAX_GAME_LENGTH)
        plane_tensor = plane_tensor.unsqueeze(0)
        output = model(plane_tensor)
        pred = output.item()
        return {"elo": int(round(pred)), "success": True}
        
    except Exception e:
        return {"elo": 0, "success": False}
