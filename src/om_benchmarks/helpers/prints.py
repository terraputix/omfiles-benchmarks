from omfiles.types import BasicSelection
from zarr.core.buffer import NDArrayLike


def print_data_info(data: NDArrayLike, chunk_size: BasicSelection) -> None:
    print(
        f"""
        Data shape: {data.shape}
        Data type: {data.dtype}
        Chunk size: {chunk_size}
        """
    )
