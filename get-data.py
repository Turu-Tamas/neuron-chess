import argparse
from itertools import batched
from process_pgns import *
import multiprocessing as mp
import numpy as np
import urllib.request
import zipfile
import gzip
import os
import subprocess
import onnx
from tqdm import tqdm

def download_and_extract_pgn(date_str: str, output_dir: str = "data/raw"):
    """Download and extract PGN file from nikonoel database."""
    os.makedirs(output_dir, exist_ok=True)
    
    url = f"https://database.nikonoel.fr/lichess_elite_{date_str}.zip"
    zip_path = os.path.join(output_dir, f"lichess_elite_{date_str}.zip")
    
    print(f"Downloading {url}...")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
    else:
        print(f"{zip_path} already exists, skipping download")
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted to {output_dir}")
    
    return os.path.join(output_dir, f"lichess_elite_{date_str}.pgn")

def download_and_prepare_weights(weights_dir: str = "weights"):
    """Download model weights and convert to ONNX with hidden representation extracted."""
    os.makedirs(weights_dir, exist_ok=True)
    
    url = "https://github.com/CallOn84/LeelaNets/raw/refs/heads/main/Nets/Maia%202200/maia-2200.pb.gz"
    gz_path = os.path.join(weights_dir, "maia-2200.pb.gz")
    pb_path = os.path.join(weights_dir, "maia-2200.pb")
    onnx_path = os.path.join(weights_dir, "maia-2200.onnx")
    hidden_path = os.path.join(weights_dir, "maia-2200-hidden.onnx")
    
    # Download
    print(f"Downloading {url}...")
    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(url, gz_path)
        print(f"Downloaded to {gz_path}")
    else:
        print(f"{gz_path} already exists, skipping download")
    
    # Extract gzip
    print(f"Extracting {gz_path}...")
    if not os.path.exists(pb_path):
        with gzip.open(gz_path, 'rb') as f_in:
            with open(pb_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"Extracted to {pb_path}")
    else:
        print(f"{pb_path} already exists, skipping extraction")
    
    # Convert to ONNX using lc0
    print(f"Converting {pb_path} to ONNX...")
    if not os.path.exists(onnx_path):
        subprocess.run(["lc0", "leela2onnx", f"--input={pb_path}", f"--output={onnx_path}"], check=True)
        print(f"Converted to {onnx_path}")
    else:
        print(f"{onnx_path} already exists, skipping conversion")
    
    # Extract hidden representation
    print(f"Extracting hidden representation from {onnx_path}...")
    if not os.path.exists(hidden_path):
        onnx.utils.extract_model(
            onnx_path,
            hidden_path,
            ["/input/planes"],
            ["/block5/conv2/relu"]
        )
        print(f"Extracted hidden representation to {hidden_path}")
    else:
        print(f"{hidden_path} already exists, skipping extraction")
    
    return hidden_path

def main():
    parser = argparse.ArgumentParser(description="Download and process chess games with neural network")
    parser.add_argument("--num-games", type=int, default=1000, help="Number of games to process")
    parser.add_argument("--positions-per-game", type=int, default=10, help="Number of positions to extract per game")
    parser.add_argument("--min-plies", type=int, default=None, help="Minimum number of plies per game (defaults to positions-per-game)")
    parser.add_argument("--with-history", action="store_true", help="Include move history in inputs")
    parser.add_argument("--output-dtype", type=str, default="float16", choices=["float16", "float32"], help="Data type for output")
    parser.add_argument("--num-processes", type=int, default=8, help="Total number of worker processes")
    parser.add_argument("--date", type=str, default="2025-11", help="Date for PGN download (format: YYYY-MM)")
    parser.add_argument("--input-file", type=str, help="Input PGN file (auto-determined if not specified)")
    parser.add_argument("--output-file", type=str, default="data/lc0-hidden/lichess_elite_2025-11.h5", help="Output HDF5 file")
    parser.add_argument("--compression", type=str, default=None, choices=[None, "gzip", "lzf"], help="Compression algorithm for HDF5 datasets (gzip or lzf)")

    args = parser.parse_args()

    # Set min_plies to positions_per_game if not specified
    if args.min_plies is None:
        args.min_plies = args.positions_per_game
    else:
        args.min_plies = min(args.min_plies, args.positions_per_game)
    
    # Download and prepare weights and PGN
    print("=" * 60)
    print("Preparing model weights...")
    print("=" * 60)
    download_and_prepare_weights()
    
    print("\n" + "=" * 60)
    print("Downloading PGN file...")
    print("=" * 60)
    pgn_file = download_and_extract_pgn(args.date)
    
    # Use provided input file if specified
    if args.input_file is not None:
        pgn_file = args.input_file

    dtype_map = {"float16": np.float16, "float32": np.float32}
    output_dtype = dtype_map[args.output_dtype]

    NUM_MODELS = args.num_processes // 4
    NUM_PREPROCESSING = NUM_MODELS * 3
    QSIZE = NUM_MODELS + 1

    print("\n" + "=" * 60)
    print("Processing Configuration:")
    print("=" * 60)
    print(f"  Input file: {pgn_file}")
    print(f"  Output file: {args.output_file}")
    print(f"  Num games: {args.num_games}")
    print(f"  Positions per game: {args.positions_per_game}")
    print(f"  Min plies: {args.min_plies}")
    print(f"  With history: {args.with_history}")
    print(f"  Output dtype: {args.output_dtype}")
    print(f"  Total processes: {args.num_processes}")
    print(f"  Computation processes: {NUM_MODELS}")
    print(f"  Preprocessing processes: {NUM_PREPROCESSING}")
    print(f"  Compression: {args.compression or 'none'}")
    print("=" * 60 + "\n")

    os.makedirs("/".join(args.output_file.split("/")[:-1]), exist_ok=True)
    mp.set_start_method("spawn", force=True)

    moves_queue = mp.Queue(maxsize=QSIZE)
    input_queue = mp.Queue(maxsize=QSIZE)
    output_queue = mp.Queue(maxsize=QSIZE)
    metadata_queue = mp.Queue(maxsize=QSIZE)

    # Create processes
    computers = []
    for _ in range(NUM_MODELS):
        computers.append(
            mp.Process(target=computation_process_main, args=(input_queue, output_queue, "weights/maia-2200-hidden.onnx"))
        )

    writer = mp.Process(
        target=write_process_main, 
        args=(output_queue, metadata_queue, args.output_file), 
        kwargs={"dtype": output_dtype, "compression": args.compression}
    )

    preprocessors = []
    for _ in range(NUM_PREPROCESSING):
        preprocessors.append(
            mp.Process(
                target=prepare_inputs_main, 
                args=(moves_queue, input_queue, args.positions_per_game), 
                kwargs={"no_history": not args.with_history}
            )
        )

    # Start processes
    for preproc in preprocessors:
        preproc.start()
    for computer in computers:
        computer.start()
    writer.start()

    # Process games
    limit_reached = False
    moves_iter = batched(read_pgn_iter(pgn_file, min_plies=args.min_plies), 100)
    games_processed = 0
    with tqdm(total=args.num_games, desc="Processing games") as pbar:
        for idx, games in enumerate(moves_iter):
            meta, moves = zip(*games)

            num_to_process = min(len(games), args.num_games - games_processed)
            if num_to_process <= 0:
                limit_reached = True
                break
            meta = meta[:num_to_process]
            moves = moves[:num_to_process]
            metadata_queue.put(meta)
            moves_queue.put((moves, idx))
            games_processed += num_to_process
            pbar.update(num_to_process)

    if not limit_reached:
        print("All games read from file.")

    metadata_queue.put("end")

    # Shutdown processes
    for preproc in preprocessors:
        moves_queue.put("end")
    for preproc in preprocessors:
        preproc.join()
    moves_queue.close()

    for compute in computers:
        input_queue.put("end")
    for compute in computers:
        compute.join()
    input_queue.close()

    output_queue.put("end")
    writer.join()
    metadata_queue.close()
    output_queue.close()

    moves_queue.join_thread()
    input_queue.join_thread()
    metadata_queue.join_thread()
    output_queue.join_thread()

    print("Processing complete!")

if __name__ == "__main__":
    main()