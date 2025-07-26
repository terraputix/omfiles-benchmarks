import random


def random_indices_for_read_range(
    read_range: tuple[int, int, int], data_shape: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    """
    Generate a random range for a given read_range
    """
    # For each dimension, pick a random start so that start + length <= dim_size
    starts = [random.randint(0, dim_size - req_len) for dim_size, req_len in zip(data_shape, read_range)]
    s0 = slice(starts[0], starts[0] + read_range[0])
    s1 = slice(starts[1], starts[1] + read_range[1])
    s2 = slice(starts[2], starts[2] + read_range[2])
    return (s0, s1, s2)


def generate_read_indices(
    data_shape: tuple[int, int, int], read_iterations: int, read_ranges: list[tuple[int, int, int]]
) -> dict[tuple[int, int, int], list[tuple[slice, slice, slice]]]:
    """
    Generate read_indices: for each read_range, generate read_iterations tuples of slices
    """
    read_indices: dict[tuple[int, int, int], list[tuple[slice, slice, slice]]] = {}
    for read_range in read_ranges:
        slices: list[tuple[slice, slice, slice]] = []
        for _ in range(read_iterations):
            slices.append(random_indices_for_read_range(read_range, data_shape))
        read_indices[read_range] = slices

    return read_indices
