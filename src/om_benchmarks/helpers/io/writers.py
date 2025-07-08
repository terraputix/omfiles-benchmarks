"""Implements a unified interface for writing data to various formats."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import h5py
import netCDF4 as nc
import numpy as np
import omfiles as om
import zarr
from zarr.core.buffer import NDArrayLike

from om_benchmarks.helpers.io.writer_configs import FormatWriterConfig, HDF5Config, NetCDFConfig, OMConfig, ZarrConfig

ConfigType = TypeVar("ConfigType", bound=FormatWriterConfig)


class BaseWriter(ABC, Generic[ConfigType]):
    config: ConfigType

    def __init__(self, filename: str, config: ConfigType):
        self.filename = Path(filename)
        self.config = config

    @abstractmethod
    def write(self, data: NDArrayLike) -> None:
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


class HDF5Writer(BaseWriter[HDF5Config]):
    def write(self, data: NDArrayLike) -> None:
        with h5py.File(self.filename, "w") as f:
            if self.config.explicitly_convert_to_int:
                # Fixme: This is just hardcoded for now.
                # Ideally we would be using this from netcdf4, so that we can specify the correct
                # scaling factor and offset converting to int32
                data = (data * 100).astype(np.int32)  # type: ignore

            f.create_dataset(
                "dataset",
                data=data,
                chunks=self.config.chunk_size,
                compression=self.config.compression,
                compression_opts=self.config.compression_opts,
                scaleoffset=self.config.scale_offset,
            )


class ZarrWriter(BaseWriter[ZarrConfig]):
    def write(self, data: NDArrayLike) -> None:
        root = zarr.open(str(self.filename), mode="w", zarr_format=self.config.zarr_format)
        # Ensure root is a Group and not an Array (for type checker)
        if not isinstance(root, zarr.Group):
            raise TypeError("Expected root to be a zarr.hierarchy.Group")
        arr_0 = root.create_array(
            "arr_0",
            shape=data.shape,
            chunks=self.config.chunk_size,
            dtype=self.config.dtype,
            compressors=self.config.compressor,
            filters=self.config.filter,
            serializer=self.config.serializer,
        )
        arr_0[:] = data


class NetCDFWriter(BaseWriter[NetCDFConfig]):
    def write(self, data: NDArrayLike) -> None:
        with nc.Dataset(self.filename, "w", format="NETCDF4") as ds:
            print("nc.__has_szip_support__", nc.__has_szip_support__)
            print("nc.__has_blosc_support__", nc.__has_blosc_support__)
            print("ds.has_szip_filter()", ds.has_szip_filter())
            dimension_names = tuple(f"dim{i}" for i in range(data.ndim))
            for dim, size in zip(dimension_names, data.shape):
                ds.createDimension(dim, size)

            var = ds.createVariable(
                varname="dataset",
                datatype=data.dtype,
                dimensions=dimension_names,
                compression=self.config.compression,
                complevel=self.config.compression_level,
                chunksizes=self.config.chunk_size,
                szip_coding="nn",
                szip_pixels_per_block=32,
                significant_digits=self.config.significant_digits,
            )
            var.scale_factor = self.config.scale_factor
            var.add_offset = self.config.add_offset
            var[:] = data

            print("ds.variables['dataset'].filters()", ds.variables["dataset"].filters())


class OMWriter(BaseWriter[OMConfig]):
    def write(self, data: NDArrayLike) -> None:
        writer = om.OmFilePyWriter(str(self.filename))
        variable = writer.write_array(
            data=data.__array__(),
            chunks=self.config.chunk_size,
            scale_factor=self.config.scale_factor,
            add_offset=self.config.add_offset,
            compression=self.config.compression,
        )
        writer.close(variable)
