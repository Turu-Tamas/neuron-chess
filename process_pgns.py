import os
import multiprocessing as mp
import chess
from lczero.backends import GameState, Input, Output, Weights, BackendCapabilities, Backend
import chess.pgn as pgn
import urllib.request
import gzip
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import onnxruntime as ort
import h5py
from functools import partial
import sys
import numpy.ma

BATCH_SIZE = 1000
BLACK_WIN = -1
WHITE_WIN = 1
DRAW = 0

TERMINATION_TO_ID = {
    # Normal endings
    "Normal": 1,
    "Checkmate": 2,
    "White resigned": 3,
    "Black resigned": 4,
    "White resigns": 3,
    "Black resigns": 4,

    # Draws
    "Stalemate": 10,
    "Draw agreed": 11,
    "Agreement": 11,
    "Draw by repetition": 12,
    "Draw by threefold repetition": 12,
    "Draw by fifty-move rule": 13,
    "Draw by 50-move rule": 13,
    "Draw by insufficient material": 14,
    "Insufficient material": 14,

    # Time-related
    "Time forfeit": 20,
    "White ran out of time": 21,
    "Black ran out of time": 22,
    "Draw by timeout vs insufficient material": 23,

    # Administrative / external
    "Forfeit": 30,
    "Adjudication": 31,
    "Game abandoned": 32,

    # Technical / online
    "Connection lost": 40,
    "Server error": 41,
    "Technical issue": 42,
}

@dataclass
class ChessMetadata:
    white_id: np.ndarray
    black_id: np.ndarray
    white_elo: np.ndarray
    black_elo: np.ndarray
    result: np.ndarray
    # openings have subtypes, primary type is first, subtypes after
    opening_ids: np.ma.MaskedArray
    time_control: np.ndarray
    white_rating_diff: np.ndarray
    black_rating_diff: np.ndarray
    termination: np.ndarray

    @staticmethod
    def with_preallocated_size(n: int):
        return ChessMetadata(
            white_elo=np.empty([n], dtype=np.uint16),
            black_elo=np.empty([n], dtype=np.uint16),
            white_id=np.empty([n], dtype=np.uint32),
            black_id=np.empty([n], dtype=np.uint32),
            result=np.empty([n], dtype=np.int8),
            opening_ids=np.ma.empty([n, 4], dtype=np.uint16),
            time_control=np.empty([n], dtype="U9"),
            white_rating_diff=np.empty([n], dtype=np.uint16),
            black_rating_diff=np.empty([n], dtype=np.uint16),
            termination=np.empty([n], dtype=np.uint8)
        )

def read_pgn(file_path: str):
    BUFFER_SIZE = 5000
    metadatas = []
    buffer = ChessMetadata.with_preallocated_size(BUFFER_SIZE)
    buffer_idx = 0
    player_map = {}
    opening_map = {}

    def player_id(name):
        if name not in player_map.keys():
            player_map[name] = len(player_map)
        return player_map[name]

    def opening_ids(opening_names, op_map=opening_map):
        name = opening_names[0]
        if name not in op_map.keys():
            op_map[name] = (len(op_map), {})
        val = op_map[name]
        return val[0] + (opening_ids(name[1:], val[1]) if len(opening_names) > 1 else [])
    
    with open(file_path, 'r') as file:
        game = pgn.read_game(file)
        while game is not None:
            buffer.white_id[buffer_idx] = player_id(game.headers["White"])
            buffer.black_id[buffer_idx] = player_id(game.headers["Black"])
            buffer.white_elo[buffer_idx] = int(game.headers["WhiteElo"])
            buffer.black_elo[buffer_idx] = int(game.headers["BlackElo"])
            result = game.headers["Result"]
            buffer.result[buffer_idx] = -1 if result == "0-1" else 1 if result == "1-0" else 0
            buffer.time_control[buffer_idx] = game.headers["TimeControl"]
            buffer.white_rating_diff[buffer_idx] = game.headers["WhiteRatingDiff"]
            buffer.black_rating_diff[buffer_idx] = game.headers["BlackRatingDiff"]
            buffer.termination[buffer_idx] = TERMINATION_TO_ID[game.headers["Termination"]]
            names = game.headers["Opening"].split(", ")
            all_names = names[0].split(": ") + names[1:]
            ids = opening_ids(all_names)
            buffer.opening_ids[buffer_idx][:len(ids)] = ids

            buffer_idx += 1
            if buffer_idx >= BUFFER_SIZE:
                metadatas.append(buffer)
                buffer = ChessMetadata.with_preallocated_size(BUFFER_SIZE)
                buffer_idx = 0
            game = pgn.read_game(file)

def download_weights():
    URL = "https://github.com/CallOn84/LeelaNets/raw/refs/heads/main/Nets/Maia%202200/maia-2200.pb.gz"
    local_path = "weights/maia-2200.pb.gz"
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(URL, local_path)
    if not os.path.exists(local_path[:-3]):
        with gzip.open(local_path, 'rb') as f_in:
            with open(local_path[:-3], 'wb') as f_out:
                f_out.write(f_in.read())
    return local_path[:-3]

def reading_process_main(queue: mp.Queue, file_path: str):
    with open(file_path) as file:
        n_games = 0
        game = pgn.read_game(file)
        while game is not None:
            queue.put(list(game.mainline_moves()))
            n_games += 1
            if n_games % 1000 == 0:
                print(f"{n_games} games read")
            game = pgn.read_game(file)
    
    print("reader end")
    queue.put("end")

def states_from_moves(moves: list[chess.Move]):
    no_history = []
    with_history = []
    board = chess.Board()
    uci_moves = []
    for move in moves:
        board.push(move)
        uci_moves.append(move.uci())
        with_history.append(GameState(moves=uci_moves))
        no_history.append(GameState(fen=board.fen()))
    return with_history, no_history

def write_prcess_main(output_queue: mp.Queue, metadata_queue: mp.Queue, out_file_path: str):
    out_file = h5py.File(out_file_path, "w")
    dset_with = out_file.create_dataset("with_history", shape=[0, 0, 0, 0], dtype=float, chunks=True, maxshape=(None, 112, 8, 8))
    dset_without = out_file.create_dataset("no_history", shape=[0, 0, 0, 0],  dtype=float, chunks=True, maxshape=(None, 112, 8, 8))
    def append_file(model_out, dset: h5py.Dataset):
        model_out = np.array(model_out)
        dset.resize(dset.shape[0] + model_out.shape[0], axis=0)
        dset[-model_out.shape[0]:] = model_out

    out_file.close()

def computation_process_main(in_queue: mp.Queue, out_queue: mp.Queue, weights_path: str):
    ort_session = ort.InferenceSession(weights_path)
    output_name = ort_session.get_outputs()[0].name
    input_name = ort_session.get_inputs()[0].name

    weights = Weights("weights/maia-2200.pb")
    backend = Backend(weights, "random")
    
    inputs_with_history = []
    inputs_no_history = []

    moves = in_queue.get()
    state_to_arr = lambda state: np.array(state.as_input(backend).GetRawInputOnnx(), dtype=np.float32).reshape(112, 8, 8)
    positions_evaluated = 0
    eval100000 = 0

    while moves != "end":
        with_history, no_history = states_from_moves(moves)
        inputs_with_history.extend(map(state_to_arr, with_history))
        inputs_no_history.extend(map(state_to_arr, no_history))

        if (len(inputs_with_history) > BATCH_SIZE):
            res_with = ort_session.run(
                [output_name],
                {input_name: np.stack(inputs_with_history)})
            res_without = ort_session.run(
                [output_name],
                {input_name: np.stack(inputs_no_history)})
            out_queue.put((res_with, res_without))
            positions_evaluated += 2*len(inputs_with_history)

            inputs_no_history.clear()
            inputs_with_history.clear()

        if positions_evaluated > 100_000:
            eval100000 += 1
            print(f"evaluated over {eval100000} * 1e5 postions")
            positions_evaluated -= 100_000

        moves = in_queue.get()


def process_file(file_path: str):
    fname = file_path.split("/")[-1][:-3]
    queue = mp.Queue(maxsize=2*BATCH_SIZE)
    reader = mp.Process(target=reading_process_main, args=[queue, file_path])
    computer = mp.Process(target=computation_process_main, args=[queue, "weights/maia-2200-hidden.onnx", f"data/lc0-hidden/maia-2200/{fname}h5"])

    reader.start()
    computer.start()
    computer.join()
    reader.join()

def main():
    download_weights()
    process_file("data/raw/lichess_elite_2025-11.pgn")

if __name__ == "__main__":
    main()