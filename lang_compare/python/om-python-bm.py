import argparse
import time

from omfiles import OmFilePyReader


def main():
    parser = argparse.ArgumentParser(description="OM File Python Benchmark")
    parser.add_argument("FILE", help="Path to OM file")
    parser.add_argument("T", nargs="?", help="T dimension to read")
    parser.add_argument("Y", nargs="?", help="Y dimension to read")
    parser.add_argument("X", nargs="?", help="X dimension to read")
    parser.add_argument("ITERATIONS", nargs="?", help="Number of iterations")
    args = parser.parse_args()

    file_path = args.FILE

    om_file = OmFilePyReader.from_path(file_path)
    dims = om_file.shape

    t = dims[0]
    y = dims[1]
    x = dims[2]
    iterations = 1

    if args.T is not None:
        try:
            t = int(args.T)
        except Exception:
            t = dims[0]
    if args.Y is not None:
        try:
            y = int(args.Y)
        except Exception:
            y = dims[1]
    if args.X is not None:
        try:
            x = int(args.X)
        except Exception:
            x = dims[2]
    if args.ITERATIONS is not None:
        try:
            iterations = int(args.ITERATIONS)
        except Exception:
            iterations = 1

    read_range = (
        slice(0, t),
        slice(0, y),
        slice(0, x),
    )

    _data_len = 0
    for _ in range(iterations):
        start = time.time()
        data = om_file[read_range]
        _data_len = len(data)
        elapsed = time.time() - start
        print("{:.6f}".format(elapsed))


if __name__ == "__main__":
    main()
