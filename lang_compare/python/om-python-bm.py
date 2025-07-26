import argparse
import time

from omfiles import OmFilePyReader


def main():
    parser = argparse.ArgumentParser(description="OM File Python Benchmark")
    parser.add_argument("FILE", help="Path to OM file")
    parser.add_argument("X", type=int, help="X dimension to read (required)")
    parser.add_argument("Y", type=int, help="Y dimension to read (required)")
    parser.add_argument("T", type=int, help="T dimension to read (required)")
    parser.add_argument("ITERATIONS", type=int, help="Number of iterations to run (required)")
    args = parser.parse_args()

    file_path = args.FILE
    x = args.X
    y = args.Y
    t = args.T
    iterations = args.ITERATIONS

    om_file = OmFilePyReader.from_path(file_path)
    dims = om_file.shape

    x_max = dims[0] - x if dims[0] > x else 0
    y_max = dims[1] - y if dims[1] > y else 0
    t_max = dims[2] - t if dims[2] > t else 0

    read_selections = []
    for i in range(iterations):
        x_start = 0 if x_max == 0 else i % x_max
        y_start = 0 if y_max == 0 else i % y_max
        t_start = 0 if t_max == 0 else i % t_max
        read_selections.append(
            (
                slice(x_start, x_start + x),
                slice(y_start, y_start + y),
                slice(t_start, t_start + t),
            )
        )

    _data_len = 0
    for read_range in read_selections:
        start = time.time()
        data = om_file[read_range]
        elapsed = time.time() - start
        _data_len = len(data)
        print("{:.6f}".format(elapsed))


if __name__ == "__main__":
    main()
