"""Implements a unified interface for reading data from various formats."""

import os
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Literal, Optional, cast

import h5py
import netCDF4 as nc
import numpy as np
import omfiles as om
import tensorstore as ts
import xarray as xr
import zarr
from omfiles.types import BasicSelection
from zarr.storage import LocalStore

# from om_benchmarks.io.MemoryMappedStore import MemoryMappedStore


class BaseReader(ABC):
    filename: Path

    def __init__(self, filename: str):
        self.filename = Path(filename)

    @classmethod
    async def create(cls, filename: str):
        return cls(filename)

    @abstractmethod
    async def read(self, index: BasicSelection) -> np.ndarray:
        raise NotImplementedError("The read method must be implemented by subclasses")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError("The close method must be implemented by subclasses")

    @abstractproperty
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError("The shape property must be implemented by subclasses")

    @abstractproperty
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        raise NotImplementedError("The chunk_shape property must be implemented by subclasses")

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


class BaselineReader(BaseReader):
    """
    Baseline reader for numpy arrays serialized via numpy's .npy format using mmap.
    """

    _mmap_array: Optional[np.memmap] = None
    _shape: Optional[tuple[int, ...]] = None
    _dtype: Optional[np.dtype] = None

    @classmethod
    async def create(cls, filename: str):
        self = await super().create(filename)
        # Use numpy memmap to open the file in read-only mode
        self._mmap_array = cast(np.memmap, np.lib.format.open_memmap(self.filename, mode="r"))
        self._shape = self._mmap_array.shape
        self._dtype = self._mmap_array.dtype
        return self

    async def read(self, index: BasicSelection) -> np.ndarray:
        if self._mmap_array is None:
            raise RuntimeError("File not opened")

        selected = self._mmap_array[index]
        # self._mmap_array._mmap.madvise(mmap.MADV_DONTNEED)  # type: ignore
        selected = np.copy(selected)
        assert selected.base is None, "Array is a view, but should be a copy"
        return selected

    def close(self) -> None:
        # np.memmap does not require explicit close, but we can delete the reference
        self._mmap_array = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            raise RuntimeError("Shape not available")
        return self._shape

    @property
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        # No chunking for baseline
        return None


class HDF5Reader(BaseReader):
    h5_reader: h5py.Dataset

    @classmethod
    async def create(cls, filename: str) -> "HDF5Reader":
        self = await super().create(filename)
        # Disable chunk caching by setting cache properties
        # Parameters: (chunk_cache_mem_size, chunk_cache_nslots, chunk_cache_w0)
        # Setting size to 0 effectively disables the cache
        # https://docs.h5py.org/en/stable/high/file.html#chunk-cache
        file = h5py.File(self.filename, "r", rdcc_nbytes=0, rdcc_nslots=0, rdcc_w0=0)
        dataset = file["dataset"]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError("Expected a h5py Dataset")
        self.h5_reader = dataset
        return self

    async def read(self, index: BasicSelection) -> np.ndarray:
        return self.h5_reader[index]

    def close(self) -> None:
        self.h5_reader.file.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.h5_reader.shape

    @property
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        return self.h5_reader.chunks


class HDF5HidefixReader(BaseReader):
    h5_reader: xr.Dataset

    @classmethod
    async def create(cls, filename: str):
        self = await super().create(filename)
        self.h5_reader = xr.open_dataset(self.filename, engine="hidefix")
        return self

    async def read(self, index: BasicSelection) -> np.ndarray:
        # hidefix arrays need to be squeezed explicitly
        return self.h5_reader["dataset"].squeeze().values

    def close(self) -> None:
        self.h5_reader.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.h5_reader["dataset"].shape

    @property
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        return cast(Optional[tuple[int, ...]], self.h5_reader["dataset"].chunks)


class ZarrReader(BaseReader):
    zarr_reader: zarr.Array

    @classmethod
    async def create(cls, filename: str):
        store = LocalStore(filename)
        # store = MemoryMappedStore(filename)
        self = await super().create(filename)
        z = zarr.open(store, mode="r")
        if not isinstance(z, zarr.Group):
            raise TypeError("Expected a zarr Group")
        array = z["arr_0"]
        if not isinstance(array, zarr.Array):
            raise TypeError("Expected a zarr Array")

        self.zarr_reader = array
        return self

    async def read(self, index: BasicSelection) -> np.ndarray:
        return self.zarr_reader[index].__array__()

    def close(self) -> None:
        self.zarr_reader.store.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.zarr_reader.shape

    @property
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        return self.zarr_reader.chunks


class ZarrsCodecsZarrReader(ZarrReader):
    zarr_reader: zarr.Array

    @classmethod
    async def create(cls, filename: str):
        import zarrs  # noqa: F401

        zarr.config.set(
            {
                # "threading.num_workers": None,
                # "array.write_empty_chunks": False,
                "codec_pipeline": {
                    "path": "zarrs.ZarrsCodecPipeline",
                    # "validate_checksums": True,
                    # "store_empty_chunks": False,
                    # "chunk_concurrent_minimum": 4,
                    # "chunk_concurrent_maximum": 1,
                    "batch_size": 1,
                }
            }
        )
        self = await super().create(filename)
        return self

    def close(self) -> None:
        zarr.config.reset()  # reset the config to its original state
        super().close()


class TensorStoreZarrReader(BaseReader):
    ts_reader: ts.TensorStore  # type: ignore

    @classmethod
    async def create(cls, filename: str):
        self = await super().create(filename)
        # Open the Zarr file using TensorStore
        self.ts_reader = await ts.open(  # type: ignore
            {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": str(self.filename),
                },
                "path": "arr_0",
                "open": True,
            }
        )

        return self

    async def read(self, index: BasicSelection) -> np.ndarray:
        return self.ts_reader[index].read().result()

    def close(self) -> None:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self.ts_reader.shape

    @property
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        return self.ts_reader.chunk_layout.read_chunk.shape


class NetCDFReader(BaseReader):
    nc_reader: nc.Dataset

    @classmethod
    async def create(cls, filename: str):
        self = await super().create(filename)
        # disable netcdf caching: https://www.unidata.ucar.edu/software/netcdf/workshops/2012/nc4chunking/Cache.html
        nc.set_chunk_cache(0, 0, 0)

        self.nc_reader = nc.Dataset(self.filename, "r")

        return self

    async def read(self, index: BasicSelection) -> np.ndarray:
        return self.nc_reader.variables["dataset"][index]

    def close(self) -> None:
        self.nc_reader.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.nc_reader.variables["dataset"].shape

    @property
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        chunking = self.nc_reader.variables["dataset"].chunking()
        assert chunking is not Literal, "netCDF unsupported chunking type: Literal"

        chunks = cast(tuple[int, ...], chunking)
        return chunks


class OMReader(BaseReader):
    # om_reader: om.OmFilePyReaderAsync
    om_reader: om.OmFilePyReader

    @classmethod
    async def create(cls, filename: str):
        self = await super().create(filename)
        # self.om_reader = await om.OmFilePyReaderAsync.from_path(str(self.filename))
        self.om_reader = om.OmFilePyReader.from_path(str(self.filename))
        return self

    async def read(self, index: BasicSelection) -> np.ndarray:
        # return await self.om_reader.read_concurrent(index)
        return self.om_reader[index]

    def close(self) -> None:
        self.om_reader.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.om_reader.shape

    @property
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        return self.om_reader.chunks
