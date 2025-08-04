"""Implements a unified interface for writing data to various formats, including a baseline mmap writer."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import h5py
import netCDF4 as nc
import numpy as np
import omfiles as om
import zarr
import zarr.storage

from om_benchmarks.io.writer_configs import (
    BaselineConfig,
    FormatWriterConfig,
    HDF5Config,
    NetCDFConfig,
    OMConfig,
    XBitInfoZarrConfig,
    ZarrConfig,
)

ConfigType = TypeVar("ConfigType", bound=FormatWriterConfig)


class BaseWriter(ABC, Generic[ConfigType]):
    config: ConfigType

    def __init__(self, filename: str, config: ConfigType):
        self.filename = Path(filename)
        self.config = config

    @abstractmethod
    def write(self, data: np.ndarray) -> None:
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


class BaselineWriter(BaseWriter[BaselineConfig]):
    """
    Baseline writer that serializes a numpy array to disk using numpy's memmap
    """

    def write(self, data: np.ndarray) -> None:
        assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
        # Write data via memmap
        mm = np.lib.format.open_memmap(self.filename, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data[:]
        mm.flush()
        del mm


class HDF5Writer(BaseWriter[HDF5Config]):
    def write(self, data: np.ndarray) -> None:
        with h5py.File(self.filename, "w") as f:
            f.create_dataset(
                "dataset",
                data=data,
                chunks=self.config.chunk_size,
                compression=self.config.compression,
                compression_opts=self.config.compression_opts,
                scaleoffset=self.config.scale_offset,
            )


class ZarrWriter(BaseWriter[ZarrConfig]):
    def write(self, data: np.ndarray) -> None:
        with zarr.storage.LocalStore(str(self.filename), read_only=False) as store:
            root = zarr.open(store, mode="w", zarr_format=self.config.zarr_format)  # type: ignore
            # Ensure root is a Group and not an Array (for type checker)
            if not isinstance(root, zarr.Group):
                raise TypeError("Expected root to be a zarr.hierarchy.Group")
            arr_0 = root.create_array(
                "arr_0",
                shape=data.shape,
                chunks=self.config.chunk_size,
                shards=self.config.shard_size,
                dtype=self.config.dtype,
                compressors=self.config.compressor,
                filters=self.config.filter,
                serializer=self.config.serializer,
            )
            arr_0[:] = data


class NetCDFWriter(BaseWriter[NetCDFConfig]):
    def write(self, data: np.ndarray) -> None:
        with nc.Dataset(self.filename, "w", format="NETCDF4") as ds:
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
                least_significant_digit=self.config.least_significant_digit,
            )
            var.scale_factor = self.config.scale_factor
            # var.add_offset = self.config.add_offset
            var[:] = data


class XbitinfoZarrWriter(BaseWriter[XBitInfoZarrConfig]):
    def write(self, data: np.ndarray) -> None:
        import xarray as xr
        from xbitinfo import get_bitinformation, get_keepbits, xr_bitround

        data_vars = {
            "dataset": (["dim0", "dim1", "dim2"], data),
        }
        ds = xr.Dataset(data_vars)

        # Compute bitinformation
        bitinfo = get_bitinformation(ds, "dim2", label="downloaded_dataset_bitinfo", implementation="julia")
        keepbits = get_keepbits(bitinfo, inflevel=self.config.information_level)

        # Bitround
        ds_bitrounded = xr_bitround(ds, keepbits)
        writer = ZarrWriter(self.filename.__str__(), self.config)
        writer.write(ds_bitrounded["dataset"])


class OMWriter(BaseWriter[OMConfig]):
    def write(self, data: np.ndarray) -> None:
        writer = om.OmFilePyWriter(str(self.filename))
        variable = writer.write_array(
            data=data.__array__(),
            chunks=self.config.chunk_size,
            scale_factor=self.config.scale_factor,
            add_offset=self.config.add_offset,
            compression=self.config.compression,
        )
        writer.close(variable)
