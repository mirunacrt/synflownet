import argparse
import random
from pathlib import Path


def parse_bool(b):
    if b.lower() in ["true", "t", "1"]:
        return True
    elif b.lower() in ["false", "f", "0"]:
        return False
    else:
        raise ValueError("Invalid boolean value")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument(
        "--filename", type=str, help="Path to input building blocks", default="short_building_blocks.txt"
    )
    parser.add_argument("--n", type=int, help="Number of building blocks to subsample", default=6000)
    parser.add_argument(
        "--random",
        type=parse_bool,
        help="If true sample building blocks uniformly at random, otherwise take the first n.",
        default=False,
    )
    args = parser.parse_args()

    with open(Path(__file__).parent / args.filename, "r") as file:
        bb_list = file.read().splitlines()

    if args.random:
        bb_list = random.sample(bb_list, args.n)
        suffix = "subsampled"
    else:
        bb_list = bb_list[: args.n]
        suffix = "first"

    new_filename = args.filename.split(".")[0] + f"_{suffix}_{args.n}.txt"
    with open(Path(__file__).parent / new_filename, "w") as file:
        for bb in bb_list:
            file.write(bb + "\n")
