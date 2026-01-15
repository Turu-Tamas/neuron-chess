import urllib.request
import argparse
import sys
import datetime
import itertools
import multiprocessing as mp
import zipfile
import os
import warnings

BASE_DIR = "data/raw/"

def file_name(year: int, month: int):
    return f"lichess_elite_{year}-{month:02d}"
def make_url(year: int, month: int):
   return f"https://database.nikonoel.fr/{file_name(year, month)}.zip"

def download_and_unzip(year: int, month: int):
    url = make_url(year, month)
    local_zip_path = f"{BASE_DIR}zips/{file_name(year, month)}.zip"
    if os.path.exists(local_zip_path):
        warnings.warn(f"Data for {year}-{month:02d} already exists, skipping download.")
        return
    else:
      try:
          urllib.request.urlretrieve(url, local_zip_path)
      except Exception as e:
          warnings.warn(f"Failed to download {url}: {e}")
          return

    if not os.path.exists(f"{BASE_DIR}{file_name(year, month)}.pgn"):
        try:
          with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
              zip_ref.extractall(BASE_DIR)
        except zipfile.BadZipFile as e:
            warnings.warn(f"Failed to download {file_name(year, month)}. The latest files are uploaded somewhat irregularly.")
            # the website redirects to an HTML page if the file is not found, this is common
            os.remove(local_zip_path) 

def main():
    end_year = datetime.datetime.now().year
    end_month = datetime.datetime.now().month

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--since", type=str, default=f"{end_year-1}-{end_month}")
    arg_parser.add_argument("--num-processes", type=int, default=mp.cpu_count())
    args = arg_parser.parse_args()

    start_year, start_month = map(int, args.since.split("-"))
    if start_year < 2020 or (start_year == 2020 and start_month < 6):
        print("Data is only available since 2020-11", file=sys.stderr)


    dates = list(itertools.chain(
        itertools.product([start_year], range(start_month, 13)),
        itertools.product(range(start_year + 1, end_year), range(1, 13)),
        itertools.product([end_year], range(1, end_month + 1))
    ))

    os.makedirs(f"{BASE_DIR}", exist_ok=True)
    os.makedirs(f"{BASE_DIR}zips/", exist_ok=True)

    with mp.Pool() as processes:
        print(f"Downloading files since {args.since} until {end_year}-{end_month:02d}")
        processes.starmap(download_and_unzip, dates)

if __name__ == "__main__":
    main()