"""Configuration classes for various data format writers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import h5py
import numcodecs.abc
from h5py._hl.filters import Gzip
from hdf5plugin import SZ, Blosc, Blosc2
from numcodecs.zarr3 import ArrayArrayCodec
from zarr.abc.codec import ArrayBytesCodec

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
    def compression_identifier(self) -> str:
        """Name and compression level"""
        raise NotImplementedError

    @property
    @abstractmethod
    def compression_pretty_name(self) -> str:
        """Pretty name of the compression"""
        raise NotImplementedError


@dataclass
class HDF5Config(FormatWriterConfig):
    """Configuration for HDF5 writer."""

    compression: Optional[Union[h5py.filters.FilterRefBase, Literal["gzip", "lzf", "szip"]]] = None
    scale_offset: Optional[int] = None
    compression_opts: Optional[tuple] = None

    @property
    def compression_identifier(self) -> str:
        """Name and compression level"""
        compression_str = ""
        if self.compression is None:
            compression_str = "none"
        elif isinstance(self.compression, str):
            compression_str = self.compression + str(self.compression_opts)
        else:
            compression_str = f"{self.compression.filter_id}_{self.compression.filter_options}"

        return f"{compression_str}_{self.scale_offset}"

    @property
    def compression_pretty_name(self) -> str:
        """Pretty name of the compression"""
        if self.compression is None:
            return "none"
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

    # Compression pipeline components
    compressor: Optional[numcodecs.abc.Codec] | Literal["auto"] = "auto"
    serializer: ArrayBytesCodec | Literal["auto"] = "auto"
    filter: ArrayArrayCodec | Literal["auto"] = "auto"

    @property
    def compression_identifier(self) -> str:
        """Name and compression level"""
        compressor_str: str = "auto"
        serializer_str: str = "auto"
        filter_str: str = "auto"
        if self.compressor != "auto" and self.compressor is not None:
            compressor_str = f"{self.compressor.codec_id}_{self.compressor.get_config()}"
        if self.serializer != "auto":
            serializer_str = f"{self.serializer.__class__.__name__}"
        if self.filter != "auto":
            filter_str = f"{self.filter.__class__.__name__}"
        return f"compressor_{compressor_str}_serializer_{serializer_str}_filter_{filter_str}"

    @property
    def compression_pretty_name(self) -> str:
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
    def compression_identifier(self) -> str:
        if self.compression is None:
            return "none"
            # return f"{self.compression}_{self.compression_level}_scale_{self.scale_factor}_offset_{str(int(self.add_offset))}_digits_{self.significant_digits}"
        return f"{self.compression}_{self.compression_level}_scale_{self.scale_factor}_digits_{self.significant_digits}"

    @property
    def compression_pretty_name(self) -> str:
        if self.compression is None:
            return "None"
        return f"{self.compression} {self.compression_level} scale {self.scale_factor} digits {self.significant_digits}"


@dataclass
class OMConfig(FormatWriterConfig):
    """Configuration for OM file writer."""

    compression: str = "pfor_delta_2d"
    scale_factor: int = 100
    add_offset: int = 0

    @property
    def compression_identifier(self) -> str:
        return f"{self.compression}_{self.scale_factor}_{self.add_offset}"

    @property
    def compression_pretty_name(self) -> str:
        if self.compression is None:
            return "None"
        return f"{self.compression} scale {self.scale_factor} offset {self.add_offset}"


@dataclass
class BaselineConfig(FormatWriterConfig):
    """Configuration for Baseline mmap numpy writer (raw .npy file)."""

    dtype: str = "f4"

    @property
    def compression_identifier(self) -> str:
        return f"baseline_{self.dtype}"

    @property
    def compression_pretty_name(self) -> str:
        return f"Baseline mmap ({self.dtype})"
