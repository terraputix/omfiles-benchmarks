from enum import Enum
from typing import Dict, Type

from .io.readers import (
    BaseReader,
    HDF5HidefixReader,
    HDF5Reader,
    NetCDFReader,
    OMReader,
    TensorStoreZarrReader,
    ZarrReader,
    ZarrsCodecsZarrReader,
)
from .io.writers import BaseWriter, HDF5Writer, NetCDFWriter, OMWriter, ZarrWriter


class AvailableFormats(Enum):
    HDF5 = "h5"
    HDF5Hidefix = "h5hidefix"
    Zarr = "zarr"
    ZarrTensorStore = "zarrTensorStore"
    ZarrPythonViaZarrsCodecs = "zarrPythonViaZarrsCodecs"
    NetCDF = "nc"
    OM = "om"

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
}

_reader_classes: Dict[AvailableFormats, Type[BaseReader]] = {
    AvailableFormats.HDF5: HDF5Reader,
    AvailableFormats.HDF5Hidefix: HDF5HidefixReader,
    AvailableFormats.Zarr: ZarrReader,
    AvailableFormats.ZarrTensorStore: TensorStoreZarrReader,
    AvailableFormats.ZarrPythonViaZarrsCodecs: ZarrsCodecsZarrReader,
    AvailableFormats.NetCDF: NetCDFReader,
    AvailableFormats.OM: OMReader,
}
