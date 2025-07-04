"""Implements a unified interface for reading data from various formats."""

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
from zarr.core.buffer.core import NDArrayLike


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
        return self.h5_reader["dataset"][index].values

    def close(self) -> None:
        self.h5_reader.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.h5_reader["dataset"].shape

    @property
    def chunk_shape(self) -> Optional[tuple[int, ...]]:
        # FIXME: unnecessary cast?
        return cast(Optional[tuple[int, ...]], self.h5_reader["dataset"].chunks)


class ZarrReader(BaseReader):
    zarr_reader: zarr.Array

    @classmethod
    async def create(cls, filename: str):
        self = await super().create(filename)
        zarr.config.set({"threading.max_workers": 8})
        z = zarr.open(str(self.filename), mode="r")
        if not isinstance(z, zarr.Group):
            raise TypeError("Expected a zarr Group")
        array = z["arr_0"]
        if not isinstance(array, zarr.Array):
            raise TypeError("Expected a zarr Array")

        self.zarr_reader = array
        return self

    async def read(self, index: BasicSelection) -> np.ndarray:
        return cast(NDArrayLike, self.zarr_reader[index]).__array__()

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
                    # "chunk_concurrent_maximum": None,
                    "batch_size": 1,
                }
            }
        )
        self = await super().create(filename)
        return self


class TensorStoreZarrReader(BaseReader):
    ts_reader: ts.TensorStore  # type: ignore

    @classmethod
    async def create(cls, filename: str):
        self = await super().create(filename)
        # Open the Zarr file using TensorStore
        self.ts_reader = ts.open(  # type: ignore
            {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": str(self.filename),
                },
                "path": "arr_0",
                "open": True,
            }
        ).result()

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
        self.om_reader = om.OmFilePyReader(str(self.filename))
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
