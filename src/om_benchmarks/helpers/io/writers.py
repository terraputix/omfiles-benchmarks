import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import h5py
import hdf5plugin
import netCDF4 as nc
import numcodecs.pcodec
import numcodecs.zarr3
import omfiles as om
import zarr
from zarr.core.buffer import NDArrayLike


class BaseWriter(ABC):
    def __init__(self, filename: str):
        self.filename = Path(filename)

    @abstractmethod
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        raise NotImplementedError("The write method must be implemented by subclasses")

    def get_file_size(self) -> int:
        """Get the size of a file in bytes."""
        path = Path(self.filename)

        # For directories (like Zarr stores), calculate total size recursively
        if path.is_dir():
            total_size = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = Path(dirpath) / f
                    if fp.is_file():
                        total_size += fp.stat().st_size
            return total_size
        # For regular files
        elif path.is_file():
            return path.stat().st_size
        else:
            return 0


class HDF5Writer(BaseWriter):
    def write(
        self,
        data: NDArrayLike,
        chunk_size: Tuple[int, ...],
        compression: str = "blosclz",
        compression_opts: int = 4,
    ) -> None:
        with h5py.File(self.filename, "w") as f:
            f.create_dataset(
                "dataset",
                data=data,
                chunks=chunk_size,
                compression=hdf5plugin.Blosc(
                    cname=compression,
                    clevel=compression_opts,
                    shuffle=hdf5plugin.Blosc.SHUFFLE,
                ),
            )


class ZarrWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        compressor = numcodecs.zarr3.Blosc(cname="lz4", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        # serializer = numcodecs.zarr3.PCodec(level=8, mode_spec="auto")
        # filter = numcodecs.zarr3.FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="i4")
        root = zarr.open(str(self.filename), mode="w", zarr_format=3)
        # Ensure root is a Group and not an Array (for type checker)
        if not isinstance(root, zarr.Group):
            raise TypeError("Expected root to be a zarr.hierarchy.Group")
        arr_0 = root.create_array(
            "arr_0",
            shape=data.shape,
            chunks=chunk_size,
            dtype="f4",
            compressors=[compressor],
            # serializer=serializer,
            # filters=[filter],
        )
        arr_0[:] = data


class NetCDFWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        with nc.Dataset(self.filename, "w", format="NETCDF4") as ds:
            dimension_names = tuple(f"dim{i}" for i in range(data.ndim))
            for dim, size in zip(dimension_names, data.shape):
                ds.createDimension(dim, size)

            var = ds.createVariable(
                varname="dataset",
                datatype=data.dtype,
                dimensions=dimension_names,
                # compression="blosc_lz",
                chunksizes=chunk_size,
            )
            var[:] = data


class OMWriter(BaseWriter):
    def write(self, data: NDArrayLike, chunk_size: Tuple[int, ...]) -> None:
        writer = om.OmFilePyWriter(str(self.filename))
        variable = writer.write_array(
            data=data.__array__(),
            chunks=chunk_size,
            scale_factor=100,
            add_offset=0,
            compression="pfor_delta_2d",
        )
        writer.close(variable)
