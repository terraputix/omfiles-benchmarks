"""Configuration classes for various data format writers."""

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import h5py
import numcodecs.abc
from zarr.core.array import FiltersLike, SerializerLike

if TYPE_CHECKING:
    from netCDF4 import CompressionLevel, CompressionType
else:
    # For runtime, define them as type aliases
    CompressionType = str
    CompressionLevel = int


@dataclass
class FormatWriterConfig(ABC):
    """Base configuration class for data format writers."""

    # Common configuration options across formats
    chunk_size: Tuple[int, ...]
    label: str = ""

    @property
    def plot_label(self) -> str:
        return self.label

    # label is not pickled -> not affecting our hash
    def __getstate__(self):
        state = self.__dict__.copy()
        if "label" in state:
            del state["label"]
        return state


@dataclass
class HDF5Config(FormatWriterConfig):
    """Configuration for HDF5 writer."""

    compression: Optional[Union[h5py.filters.FilterRefBase, Literal["gzip", "lzf", "szip"]]] = None
    # https://docs.h5py.org/en/stable/high/dataset.html#dataset-scaleoffset
    scale_offset: Optional[int] = None
    compression_opts: Optional[tuple] = None

    @property
    def absolute_tolerance(self) -> float:
        """Absolute tolerance for floating point values"""
        if self.scale_offset is None:
            return 0.0
        return 1 / (10**self.scale_offset)


@dataclass
class ZarrConfig(FormatWriterConfig):
    """Configuration for Zarr writer."""

    # Zarr-specific options
    zarr_format: int = 2
    dtype: str = "f4"
    only_python_zarr: bool = False
    shard_size: Optional[Tuple[int, ...]] = None

    # Compression pipeline components
    compressor: Optional[numcodecs.abc.Codec] | Literal["auto"] = "auto"
    serializer: SerializerLike = "auto"
    filter: FiltersLike = "auto"


@dataclass
class NetCDFConfig(FormatWriterConfig):
    """Configuration for NetCDF writer."""

    compression: Optional[CompressionType] = None
    compression_level: Optional[CompressionLevel] = 4
    # TODO: remove scale_factor
    scale_factor: float = 1.0
    # add_offset: float = 0.0
    least_significant_digit: Optional[int] = None

    @property
    def absolute_tolerance(self) -> float:
        return 10 ** (-self.least_significant_digit) if self.least_significant_digit is not None else 0.0


@dataclass
class XBitInfoZarrConfig(ZarrConfig):
    information_level: float = 0.99


@dataclass
class OMConfig(FormatWriterConfig):
    """Configuration for OM file writer."""

    compression: str = "pfor_delta_2d"
    scale_factor: int = 100
    add_offset: int = 0


@dataclass
class BaselineConfig(FormatWriterConfig):
    """Configuration for Baseline mmap numpy writer (raw .npy file)."""

    dtype: str = "f4"
