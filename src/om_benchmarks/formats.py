from enum import Enum
from typing import Dict, Type

from om_benchmarks.io.readers import (
    BaselineReader,
    BaseReader,
    HDF5HidefixReader,
    HDF5Reader,
    NetCDFReader,
    OMReader,
    TensorStoreZarrReader,
    ZarrReader,
    ZarrsCodecsZarrReader,
)
from om_benchmarks.io.writers import BaselineWriter, BaseWriter, HDF5Writer, NetCDFWriter, OMWriter, ZarrWriter


class AvailableFormats(Enum):
    HDF5 = "HDF5"
    HDF5Hidefix = "HDF5Hidefix"
    Zarr = "Zarr"
    ZarrTensorStore = "ZarrTensorStore"
    ZarrPythonViaZarrsCodecs = "ZarrPythonViaZarrsCodecs"
    NetCDF = "NetCDF"
    OM = "OM"
    Baseline = "Baseline"

    @property
    def file_extension(self) -> str:
        if self == AvailableFormats.HDF5:
            return ".h5"
        elif self == AvailableFormats.HDF5Hidefix:
            return ".h5"
        elif self == AvailableFormats.Zarr:
            return ".zarr"
        elif self == AvailableFormats.ZarrTensorStore:
            return ".zarr"
        elif self == AvailableFormats.ZarrPythonViaZarrsCodecs:
            return ".zarr"
        elif self == AvailableFormats.NetCDF:
            return ".nc"
        elif self == AvailableFormats.OM:
            return ".om"
        elif self == AvailableFormats.Baseline:
            return ".mmap"
        else:
            raise ValueError(f"Unknown format: {self.name}")

    @property
    def reader_class(self) -> Type[BaseReader]:
        """Get the reader class for this format."""
        if self not in _reader_classes:
            raise ValueError(f"No reader available for format: {self.name}")
        return _reader_classes[self]

    @property
    def writer_class(self) -> Type[BaseWriter]:
        """Get the writer class for this format, or None if writing is not supported."""
        if self not in _writer_classes:
            raise ValueError(f"No writer available for format: {self.name}")
        return _writer_classes[self]


_writer_classes: Dict[AvailableFormats, Type[BaseWriter]] = {
    AvailableFormats.HDF5: HDF5Writer,
    AvailableFormats.Zarr: ZarrWriter,
    AvailableFormats.NetCDF: NetCDFWriter,
    AvailableFormats.OM: OMWriter,
    AvailableFormats.Baseline: BaselineWriter,
}

_reader_classes: Dict[AvailableFormats, Type[BaseReader]] = {
    AvailableFormats.HDF5: HDF5Reader,
    AvailableFormats.HDF5Hidefix: HDF5HidefixReader,
    AvailableFormats.Zarr: ZarrReader,
    AvailableFormats.ZarrTensorStore: TensorStoreZarrReader,
    AvailableFormats.ZarrPythonViaZarrsCodecs: ZarrsCodecsZarrReader,
    AvailableFormats.NetCDF: NetCDFReader,
    AvailableFormats.OM: OMReader,
    AvailableFormats.Baseline: BaselineReader,
}
