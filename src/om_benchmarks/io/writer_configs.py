"""Configuration classes for various data format writers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import h5py
import numcodecs.abc
from h5py._hl.filters import Gzip
from hdf5plugin import SZ, Blosc, Blosc2
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

    @property
    @abstractmethod
    def plot_label(self) -> str:
        """Label for plotting"""
        raise NotImplementedError


@dataclass
class HDF5Config(FormatWriterConfig):
    """Configuration for HDF5 writer."""

    compression: Optional[Union[h5py.filters.FilterRefBase, Literal["gzip", "lzf", "szip"]]] = None
    scale_offset: Optional[int] = None
    compression_opts: Optional[tuple] = None

    @property
    def plot_label(self) -> str:
        """Pretty name of the compression"""
        if self.compression is None:
            return "No compression"
        elif isinstance(self.compression, str):
            return self.compression + str(self.compression_opts) + str(self.scale_offset)
        elif self.compression.__class__ is Gzip:
            return f"{self.compression.__class__.__name__} ({self.compression.filter_options[0]})"  # type: ignore
        elif self.compression.__class__ is Blosc:
            cname_no = self.compression.filter_options[6]  # type: ignore
            # Reverse map the Blosc._Blosc__COMPRESSIONS dict to get the string name from the integer code
            cname_map = {v: k for k, v in Blosc._Blosc__COMPRESSIONS.items()}  # type: ignore
            cname = cname_map.get(cname_no, str(cname_no))
            clevel = self.compression.filter_options[4]  # type: ignore
            return f"{self.compression.__class__.__name__} {cname} clevel {clevel}"
        elif self.compression.__class__ is Blosc2:
            return f"{self.compression.__class__.__name__} ({self.compression.filter_options[0]})"  # type: ignore
        elif self.compression.__class__ is SZ:
            return f"{self.compression.__class__.__name__} ({self.compression.filter_options[0]})"  # type: ignore
        else:
            raise ValueError(f"Unknown compression type: {self.compression.__class__.__name__}")


@dataclass
class ZarrConfig(FormatWriterConfig):
    """Configuration for Zarr writer."""

    # Zarr-specific options
    zarr_format: int = 2
    dtype: str = "f4"
    only_python_zarr: bool = False

    # Compression pipeline components
    compressor: Optional[numcodecs.abc.Codec] | Literal["auto"] = "auto"
    serializer: SerializerLike = "auto"
    filter: FiltersLike = "auto"

    @property
    def plot_label(self) -> str:
        compressor_str = "auto"
        serializer_str = "auto"
        filter_str = "auto"
        if self.compressor != "auto" and self.compressor is not None:
            codec_config = self.compressor.get_config()
            compressor_str = (
                f"{codec_config['id']} {codec_config.get('cname', 'auto')} {codec_config.get('clevel', 'auto')}"
            )
        if self.serializer != "auto":
            serializer_str = f"{self.serializer.__class__.__name__}"
        if self.filter != "auto":
            filter_str = f"{self.filter.__class__.__name__}"
        return f"{compressor_str} {serializer_str} {filter_str}"


@dataclass
class NetCDFConfig(FormatWriterConfig):
    """Configuration for NetCDF writer."""

    compression: Optional[CompressionType] = None
    compression_level: Optional[CompressionLevel] = 4
    scale_factor: float = 1.0
    # add_offset: float = 0.0
    significant_digits: Optional[int] = None

    @property
    def plot_label(self) -> str:
        if self.compression is None:
            return "No compression"
        return f"{self.compression} {self.compression_level} scale {self.scale_factor} digits {self.significant_digits}"


@dataclass
class OMConfig(FormatWriterConfig):
    """Configuration for OM file writer."""

    compression: str = "pfor_delta_2d"
    scale_factor: int = 100
    add_offset: int = 0

    @property
    def plot_label(self) -> str:
        return f"{self.compression} scale {self.scale_factor} offset {self.add_offset}"


@dataclass
class BaselineConfig(FormatWriterConfig):
    """Configuration for Baseline mmap numpy writer (raw .npy file)."""

    dtype: str = "f4"

    @property
    def plot_label(self) -> str:
        return "No compression"
