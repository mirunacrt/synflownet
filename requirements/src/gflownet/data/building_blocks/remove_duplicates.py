import argparse


def remove_duplicates(building_blocks: list[str]) -> list[str]:
    return list(set(building_blocks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicates from building blocks")
    parser.add_argument("--building_blocks_filename", type=str, help="Path to building blocks")
    parser.add_argument("--output_filename", type=str, help="Path to output building blocks without duplicates")
    args = parser.parse_args()

    with open(args.building_blocks_filename, "r") as file:
        building_blocks = file.read().splitlines()

    print("Removing duplicates ...")
    building_blocks_without_duplicates = remove_duplicates(building_blocks)

    new_filename = args.output_filename
    with open(new_filename, "w") as file:
        for bb in building_blocks_without_duplicates:
            file.write(bb + "\n")
