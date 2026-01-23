import multiprocessing as mp
from lczero.backends import GameState # type: ignore
import chess.pgn as pgn
from dataclasses import dataclass
import numpy as np
import onnxruntime as ort
import h5py
from typing import Optional, Iterable
from queue import Empty
import bulletchess

INPUT_SHAPE = (112, 8, 8)
OUTPUT_SHAPE = (64, 8, 8)

BATCH_SIZE = 1000
BLACK_WIN = -1
WHITE_WIN = 1
DRAW = 0

WHITE = "White"
BLACK = "Black"
LEELA_OUTPUT = "lc0_hidden"
WHITE_RATING_DIFF = "WhiteRatingDiff"
BLACK_RATING_DIFF = "BlackRatingDiff"
WHITE_ELO = "WhiteElo"
BLACK_ELO = "BlackElo"
RESULT = "Result"
OPENING = "Opening"
TIME_CONTROL = "TimeControl"
TERMINATION = "Termination"

@dataclass
class ChessMetadata:
    white_id: int
    black_id: int
    white_elo: int
    black_elo: int
    result: int
    # openings have subtypes, primary type is first, subtypes after
    opening_name: str
    time_control: str
    white_rating_diff: Optional[int]
    black_rating_diff: Optional[int]
    termination: str

def read_pgn_iter(file_path: str, min_plies: int = 10, player_map={}, opening_map={}):
    metadatas = []

    moves = []

    def player_id(name):
        if name not in player_map:
            player_map[name] = len(player_map)
        return player_map[name]

    '''def opening_id(name, op_map=opening_map):
        if name not in op_map:
            op_map[name] = len(op_map)
        return op_map[name]'''

    def get_result(result: str):
        return BLACK_WIN if result == "0-1" else DRAW if result == "1/2-1/2" else WHITE_WIN

    with open(file_path, 'r') as file:
        while True:
            game = pgn.read_game(file)
            if game is None:
                break

            game_moves = [move.uci() for move in game.mainline_moves()]
            
            # Skip games with fewer plies than minimum
            if len(game_moves) < min_plies:
                continue

            if WHITE_RATING_DIFF in game.headers:
                white_diff = int(game.headers[WHITE_RATING_DIFF])
            else:
                white_diff = None
            if BLACK_RATING_DIFF in game.headers:
                black_diff = int(game.headers[BLACK_RATING_DIFF])
            else:
                black_diff = None

            headers = game.headers
            meta = ChessMetadata(
                white_elo=int(headers[WHITE_ELO]),
                white_id=player_id(headers[WHITE]),
                black_elo=int(headers[BLACK_ELO]),
                black_id=player_id(headers[BLACK]),
                result=get_result(headers[RESULT]),
                opening_name=headers[OPENING],
                time_control=headers[TIME_CONTROL],
                termination=headers[TERMINATION],
                white_rating_diff=white_diff,
                black_rating_diff=black_diff,
            )

            yield meta, game_moves

def metadata_structured_array(data: list[ChessMetadata]):
    n = len(data)
    
    dt = np.dtype([
        (WHITE, np.uint32),
        (BLACK, np.uint32),
        (WHITE_ELO, np.uint16),
        (BLACK_ELO, np.uint16),
        (WHITE_RATING_DIFF, np.int16),
        (BLACK_RATING_DIFF, np.int16),
        (OPENING, 'U100'),
        (TERMINATION, 'U21'),
        (RESULT, np.int8)
    ])

    result = np.zeros(n, dtype=dt)

    result[WHITE] = np.array(list(map(lambda meta: meta.white_id, data)))
    result[BLACK] = np.array(list(map(lambda meta: meta.black_id, data)))
    result[WHITE_ELO] = np.array(list(map(lambda meta: meta.white_elo, data)))
    result[BLACK_ELO] = np.array(list(map(lambda meta: meta.black_elo, data)))
    result[WHITE_RATING_DIFF] = np.array(list(map(lambda meta: meta.white_rating_diff or 0, data)))
    result[BLACK_RATING_DIFF] = np.array(list(map(lambda meta: meta.black_rating_diff or 0, data)))
    result[OPENING] = np.array(list(map(lambda meta: meta.opening_name, data)))
    result[TERMINATION] = np.array(list(map(lambda meta: meta.termination, data)), dtype='U21')
    result[RESULT] = np.array(list(map(lambda meta: meta.result, data)))

    return result

def write_process_main(output_queue: mp.Queue, metadata_queue: mp.Queue, out_file_path: str, dtype=np.float16, compression=None):
    out_file = h5py.File(out_file_path, "w")
    dsets: dict[str, h5py.Dataset] = {}

    # Board state datasets
    dsets[LEELA_OUTPUT] = out_file.create_dataset(
        LEELA_OUTPUT, shape=[0, *OUTPUT_SHAPE], dtype=dtype, 
        chunks=True, maxshape=(None, *OUTPUT_SHAPE), compression=compression
    )

    # Metadata group and datasets
    meta_group = out_file.create_group("metadata")
    dsets[WHITE_ELO] = meta_group.create_dataset(
        WHITE_ELO, shape=[0], dtype=np.uint16, 
        chunks=True, maxshape=(None,), compression=compression
    )
    dsets[BLACK_ELO] = meta_group.create_dataset(
        BLACK_ELO, shape=[0], dtype=np.uint16, 
        chunks=True, maxshape=(None,), compression=compression
    )
    dsets[WHITE_RATING_DIFF] = meta_group.create_dataset(
        WHITE_RATING_DIFF, shape=[0], dtype=np.int16, 
        chunks=True, maxshape=(None,), compression=compression
    )
    dsets[BLACK_RATING_DIFF] = meta_group.create_dataset(
        BLACK_RATING_DIFF, shape=[0], dtype=np.int16, 
        chunks=True, maxshape=(None,), compression=compression
    )
    dsets[WHITE] = meta_group.create_dataset(
        WHITE, shape=[0], dtype=np.uint32, 
        chunks=True, maxshape=(None,), compression=compression
    )
    dsets[BLACK] = meta_group.create_dataset(
        BLACK, shape=[0], dtype=np.uint32, 
        chunks=True, maxshape=(None,), compression=compression
    )
    dsets[OPENING] = meta_group.create_dataset(
        OPENING, shape=[0], dtype=h5py.string_dtype(),
        chunks=True, maxshape=(None,), compression=compression
    )
    dsets[TERMINATION] = meta_group.create_dataset(
        TERMINATION, shape=[0], dtype=h5py.string_dtype(),
        chunks=True, maxshape=(None,), compression=compression
    )
    dsets[RESULT] = meta_group.create_dataset(
        RESULT, shape=[0], dtype=np.int8,
        chunks=True, maxshape=(None,), compression=compression
    )

    def extend_dset(key: str, val: np.ndarray):
        dset = dsets[key]
        dset.resize(dset.shape[0] + val.shape[0], axis=0)
        dset[-val.shape[0]:] = val

    def process_meta(val):
        data = metadata_structured_array(val)
        for key in data.dtype.names: # type: ignore
            extend_dset(key, data[key])

    next_idx = 0
    uninserted = {}
    def process_model_out(val):
        nonlocal next_idx
        model_out, idx = val
        model_out: np.ndarray
        if idx == next_idx:
            extend_dset(LEELA_OUTPUT, model_out)
            next_idx += 1
            while next_idx in uninserted:
                extend_dset(*uninserted.pop(next_idx))
                next_idx += 1
        else:
            uninserted[idx] = (LEELA_OUTPUT, model_out)

    meta_done = False
    model_outs_done = False
    while True:
        if not meta_done:
            try:
                meta = metadata_queue.get_nowait()
            except Empty:
                meta = None
            if meta == "end":
                meta_done = True
            elif meta is not None:
                process_meta(meta)
                continue # only read from output queue if meta queue is empty

        val = output_queue.get()
        if val == "end":
            model_outs_done = True
        else:
            process_model_out(val)
        
        if model_outs_done and meta_done:
            break

    out_file.close()

def prepare_inputs(games_moves: Iterable[list[str]], positions_per_game: int, no_history: bool = False):
    all_inputs = []
    state_to_arr = lambda state: np.array(
            state.as_input_from_format(1).GetRawInputOnnx(), dtype=np.float32
        ).reshape(*INPUT_SHAPE)

    for game in games_moves:
        board = bulletchess.Board()
        uci_moves = []
        game_inputs = []
        move_idx = 0
        first_idx = len(game) % positions_per_game # (len(game) - first_idx) divisible by positions_per_game
        d = (len(game) - first_idx) // positions_per_game # distance between inputs
        for move in game:
            board.apply(bulletchess.Move.from_uci(move))
            uci_moves.append(move)
            if len(game) <= positions_per_game or move_idx == first_idx + d * len(game_inputs):
                if no_history:
                    game_inputs.append(state_to_arr(GameState(fen=board.fen())))
                else:
                    game_inputs.append(state_to_arr(GameState(moves=uci_moves)))
            move_idx += 1
        assert len(game_inputs) == positions_per_game, f"expected {positions_per_game} positions, have {len(game_inputs)}; {d}, {first_idx}"
        all_inputs.append(np.array(game_inputs))

    return np.concat(all_inputs)

def prepare_inputs_main(moves_queue: mp.Queue, input_queue: mp.Queue, positions_per_game: int, no_history=False):
    while True:
        val = moves_queue.get()
        if val == "end":
            break
        games, idx = val
        input_queue.put((prepare_inputs(games, positions_per_game, no_history=no_history), idx))

def computation_process_main(in_queue: mp.Queue, out_queue: mp.Queue, weights_path: str):
    ort_session = ort.InferenceSession(weights_path)
    output_name = ort_session.get_outputs()[0].name
    input_name = ort_session.get_inputs()[0].name

    while True:
        val = in_queue.get()
        if val == "end":
            break
        inputs, idx = val
        result = ort_session.run(
            [output_name],
            {input_name: inputs})
        out_queue.put((result[0], idx))
