"""Configuration classes for various data format writers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import h5py
import numcodecs.abc
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

    def to_string(self) -> str:
        """Convert configuration to a string representation using dataclass fields."""
        parts = []

        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                # Handle different types of values
                if isinstance(value, (tuple, list)):
                    value_str = "x".join(map(str, value))
                elif hasattr(value, "__class__") and hasattr(value, "__dict__"):
                    # For complex objects, use class name
                    value_str = value.__class__.__name__
                else:
                    value_str = str(value)

                parts.append(f"{field.name}_{value_str}")

        return "_".join(parts)


@dataclass
class HDF5Config(FormatWriterConfig):
    """Configuration for HDF5 writer."""

    compression: Optional[h5py.filters.FilterRefBase] = None

    @property
    def compression_identifier(self) -> str:
        """Name and compression level"""
        if self.compression is None:
            return "none"
        return f"{self.compression.filter_id}_{self.compression.filter_options}"


@dataclass
class ZarrConfig(FormatWriterConfig):
    """Configuration for Zarr writer."""

    # Zarr-specific options
    zarr_format: int = 2
    dtype: str = "f4"

    # Compression pipeline components
    compressor: Optional[numcodecs.abc.Codec] = None
    serializer: ArrayBytesCodec | Literal["auto"] = "auto"
    filter: numcodecs.abc.Codec | Literal["auto"] = "auto"

    @property
    def compression_identifier(self) -> str:
        """Name and compression level"""
        compressor_str: str = "none"
        serializer_str: str = "auto"
        filter_str: str = "auto"
        if self.compressor is not None:
            compressor_str = f"{self.compressor.codec_id}_{self.compressor.get_config()}"
        if self.serializer != "auto":
            serializer_str = f"{self.serializer.to_dict()}"
        if self.filter != "auto":
            filter_str = f"{self.filter.codec_id}_{self.filter.get_config()}"
        return f"compressor_{compressor_str}_serializer_{serializer_str}_filter_{filter_str}"


@dataclass
class NetCDFConfig(FormatWriterConfig):
    """Configuration for NetCDF writer."""

    compression: Optional[CompressionType] = None
    compression_level: Optional[CompressionLevel] = None

    @property
    def compression_identifier(self) -> str:
        if self.compression is None:
            return "none"
        return f"{self.compression}_{self.compression_level}"


@dataclass
class OMConfig(FormatWriterConfig):
    """Configuration for OM file writer."""

    compression: str = "pfor_delta_2d"
    scale_factor: int = 100
    add_offset: int = 0

    @property
    def compression_identifier(self) -> str:
        return f"{self.compression}_{self.scale_factor}_{self.add_offset}"
