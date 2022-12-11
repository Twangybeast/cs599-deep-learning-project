import argparse
import io

import chess
import chess.pgn
import chess.engine
from flask import Flask, request
from flask_cors import CORS, cross_origin

import torch

from main import ChessNet, game_to_tensor
from prepare_games import get_game_info

# lol run this in the src folder
FILTER_SIZE = 512
NUM_LAYERS = 30
MAX_GAME_LENGTH = 12
HIDDEN_FILTER_SIZE = 32
FINAL_HIDDEN_SIZE = 128
MODEL_PATH = "../v5_47.pt"

print("Loading model...")
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model = ChessNet(FILTER_SIZE, NUM_LAYERS, MAX_GAME_LENGTH, hidden_filter_size=HIDDEN_FILTER_SIZE, hidden_size=FINAL_HIDDEN_SIZE)
model.load_state_dict(state_dict['model'])
model.eval()

engine = chess.engine.SimpleEngine.popen_uci("../stockfish")
print("Starting app...")


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/game-elo")
@cross_origin()
def get_elo():
    try:
        pgn = request.args.get("pgn", default="", type=str)
        pgn = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn)
        game.headers['WhiteElo'] = "0"
        game.headers['BlackElo'] = "0"
        for node in game.mainline():
            info = engine.analyse(node.board(), chess.engine.Limit(time=0.2))
            node.set_eval(info['score'])
        boards, scores, elo = get_game_info(game)
        plane_tensor = game_to_tensor(boards, scores, max(0, len(boards) - MAX_GAME_LENGTH), MAX_GAME_LENGTH)
        plane_tensor = plane_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(plane_tensor)
            pred = output.item()
        return {"elo": int(round(pred)), "success": True}
        
    except Exception:
        return {"elo": 0, "success": False}

if __name__ == '__main__':
    app.run(ssl_context='adhoc', host="0.0.0.0", port="80")
