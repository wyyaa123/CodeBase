import os
import time
import argparse

parser = argparse.ArgumentParser(description="rename files")
parser.add_argument("--start", type=int, help="the begin number of filename", required=True)
parser.add_argument("--path", type=str, help="folder path", required=True)

args = parser.parse_args()

start: int = args.start
path: str = args.path


for i, file in enumerate(sorted(os.listdir(path)), start=start):
    os.rename(os.path.join(path, file), os.path.join(path, f"{i: 06d}.png"))
