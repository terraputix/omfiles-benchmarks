from functools import reduce


def _uncompressed_size_from_array_shape(array_shape: str, bytes_per_element: int = 4) -> int:
    # Remove parentheses if present, then split and convert to int
    array_shape_tuple: tuple[int, ...] = tuple(int(x) for x in array_shape.strip("()").split(",") if x.strip())
    total_size_bytes = reduce(lambda x, y: x * y, array_shape_tuple) * bytes_per_element
    return total_size_bytes
