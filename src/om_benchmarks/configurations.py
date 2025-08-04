import hashlib
import pickle
from dataclasses import replace
from typing import Dict, List, Tuple, cast

import hdf5plugin
import numcodecs
import numcodecs.zarr3
import omfiles.zarr3
from hdf5plugin import Blosc as HBlosc
from numcodecs import Blosc as NBlosc

from om_benchmarks.formats import AvailableFormats
from om_benchmarks.io.writer_configs import (
    BaselineConfig,
    FormatWriterConfig,
    HDF5Config,
    NetCDFConfig,
    OMConfig,
    XBitInfoZarrConfig,
    ZarrConfig,
)

# Global registry for configurations
CONFIG_REGISTRY: Dict[str, FormatWriterConfig] = {}


def register_config(config: FormatWriterConfig) -> str:
    """Register a configuration and return its unique identifier."""
    # Pickle can be used to also serialize non native python objects deterministically
    serialized_config = pickle.dumps(config)
    # Create a hash of the configuration's picklable attributes
    config_hash = hashlib.sha256(serialized_config).hexdigest()
    CONFIG_REGISTRY[config_hash] = config
    return config_hash


def get_config_by_hash(config_hash: str) -> FormatWriterConfig:
    """Retrieve a configuration from the registry using its hash."""
    return CONFIG_REGISTRY[config_hash]


CHUNKS = {
    "small": (5, 5, 744),
    "balanced": (32, 32, 32),
    "xtra_large": (40, 40, 744),
    # "medium": (10, 10, 744),
    # "large": (20, 20, 744),
    # "xtra_xtra_large": (100, 100, 744),
}

_BASELINE_CONFIG = BaselineConfig(chunk_size=CHUNKS["small"], label="No compression")
_NETCDF_BEST = NetCDFConfig(
    chunk_size=CHUNKS["small"],
    compression="zlib",
    compression_level=3,
    label="Zlib clevel 3",
)
_HDF5_BEST = HDF5Config(
    chunk_size=CHUNKS["small"],
    compression=HBlosc(cname="lz4", clevel=4, shuffle=HBlosc.SHUFFLE),
    label="Blosc LZ4 clevel 4, ByteS",
)
_ZARR_BEST = ZarrConfig(
    chunk_size=CHUNKS["small"],
    compressor=NBlosc(cname="lz4", clevel=4, shuffle=NBlosc.BITSHUFFLE),
    label="Blosc LZ4 clevel 4, BitS",
)
_OM_BEST = OMConfig(
    chunk_size=CHUNKS["small"],
    compression="pfor_delta_2d",
    scale_factor=20,
    add_offset=0,
    label="PFOR Delta 2D, Scale Factor 20",
)

_XBITINFO_CONFIG = XBitInfoZarrConfig(
    chunk_size=CHUNKS["small"],
    compressor=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE, blocksize=0),
    information_level=0.99,
    label="zarr2, 99% Information, Blosc LZ4 clevel 5, BitS",
)

_NETCDF_CONFIGS = [
    NetCDFConfig(chunk_size=CHUNKS["small"], compression=None, label="No Compression"),
    NetCDFConfig(chunk_size=CHUNKS["small"], compression="szip", least_significant_digit=1, label="SZIP, abs=0.1"),
    NetCDFConfig(chunk_size=CHUNKS["small"], compression="szip", least_significant_digit=2, label="SZIP, abs=0.01"),
    NetCDFConfig(chunk_size=CHUNKS["small"], compression="szip", label="SZIP"),
    _NETCDF_BEST,
]

_HDF5_CONFIGS = [
    HDF5Config(chunk_size=CHUNKS["small"], label="No compression"),  # hdf5 baseline: no compression
    _HDF5_BEST,
    # https://hdfgroup.github.io/hdf5/develop/group___s_z_i_p.html#ga688fde8106225adf9e6ccd2a168dec74
    # https://hdfgroup.github.io/hdf5/develop/_h5_d__u_g.html#title6
    # 1st 'nn' stands for: H5_SZIP_NN_OPTION_MASK
    # 2nd 32 stands for: 32 pixels per block
    HDF5Config(
        chunk_size=CHUNKS["small"],
        compression="szip",
        compression_opts=("nn", 32),
        scale_offset=2,
        label="SZIP nn 32, abs=0.01",
    ),
    # https://hdfgroup.github.io/hdf5/develop/group___s_z_i_p.html#ga688fde8106225adf9e6ccd2a168dec74
    # https://hdfgroup.github.io/hdf5/develop/_h5_d__u_g.html#title6
    # 1st 'nn' stands for: H5_SZIP_NN_OPTION_MASK
    # 2nd 32 stands for: 32 pixels per block
    HDF5Config(
        chunk_size=CHUNKS["small"],
        compression="szip",
        compression_opts=("nn", 8),
        label="SZIP nn 8",
    ),
    HDF5Config(
        chunk_size=CHUNKS["small"],
        compression=HBlosc(cname="zstd", clevel=9, shuffle=HBlosc.SHUFFLE),
        label="Blosc zstd clevel 9",
    ),
    HDF5Config(
        chunk_size=CHUNKS["small"],
        compression=hdf5plugin.SZ(absolute=0.01),
        label="SZ abs=0.01",
    ),
]

_ZARR_CONFIGS = [
    ZarrConfig(
        chunk_size=CHUNKS["small"], compressor=None, label="zarr2, No compression"
    ),  # zarr baseline: no compression
    ZarrConfig(
        chunk_size=CHUNKS["small"],
        compressor=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE, blocksize=0),  # default
        label="zarr2, Blosc LZ4 clevel 5",
    ),
    _ZARR_BEST,
    ZarrConfig(
        chunk_size=CHUNKS["small"],
        compressor=NBlosc(cname="zstd", clevel=3, shuffle=NBlosc.BITSHUFFLE),
        label="zarr2, Blosc zstd clevel 3",
    ),
    ZarrConfig(
        zarr_format=3,
        chunk_size=CHUNKS["small"],
        compressor=None,
        serializer=omfiles.zarr3.PforSerializer(),
        filter=numcodecs.zarr3.FixedScaleOffset(offset=0, scale=20, dtype="f4", astype="i4"),
        only_python_zarr=True,
        label="zarr3, PFOR, Scale Factor 20",
    ),
    ZarrConfig(
        zarr_format=3,
        chunk_size=CHUNKS["small"],
        compressor=None,
        serializer=numcodecs.zarr3.PCodec(mode_spec="auto"),  # TODO: verify this is the same
        filter=numcodecs.zarr3.FixedScaleOffset(offset=0, scale=20, dtype="f4", astype="i4"),
        only_python_zarr=True,
        label="zarr3, PCodec clevel 8, Scale Factor 20",
    ),
    ZarrConfig(
        zarr_format=3,
        chunk_size=CHUNKS["small"],
        compressor=None,
        serializer=numcodecs.zarr3.PCodec(level=8, mode_spec="auto"),
        filter=numcodecs.zarr3.FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="i4"),
        only_python_zarr=True,
        label="zarr3, PCodec clevel 8, Scale Factor 100",
    ),
    ZarrConfig(
        zarr_format=3,
        chunk_size=CHUNKS["small"],
        compressor=None,
        serializer=numcodecs.zarr3.PCodec(),
        only_python_zarr=True,
        label="zarr3, PCodec clevel 8",
    ),
]

_OM_CONFIGS = [
    _OM_BEST,
    OMConfig(
        chunk_size=CHUNKS["small"],
        compression="pfor_delta_2d",
        scale_factor=100,
        add_offset=0,
        label="PFOR Delta 2D, Scale Factor 100",
    ),
    OMConfig(
        chunk_size=CHUNKS["small"],
        compression="fpx_xor_2d",
        label="FPX XOR 2D",
    ),
]

CONFIGURATION_INVENTORY: Dict[tuple[int, int, int], List[Tuple[AvailableFormats, FormatWriterConfig]]] = {
    chunk_size: [
        (AvailableFormats.Baseline, replace(config, chunk_size=chunk_size))
        for config in [cast(FormatWriterConfig, _BASELINE_CONFIG)]
    ]
    + [(AvailableFormats.XbitInfo, replace(config, chunk_size=chunk_size)) for config in [_XBITINFO_CONFIG]]
    + [(AvailableFormats.NetCDF, replace(config, chunk_size=chunk_size)) for config in _NETCDF_CONFIGS]
    + [(AvailableFormats.HDF5, replace(config, chunk_size=chunk_size)) for config in _HDF5_CONFIGS]
    + [(AvailableFormats.Zarr, replace(config, chunk_size=chunk_size)) for config in _ZARR_CONFIGS]
    + [
        (AvailableFormats.ZarrTensorStore, replace(config, chunk_size=chunk_size))
        for config in _ZARR_CONFIGS
        if not config.only_python_zarr
    ]
    + [
        (AvailableFormats.ZarrPythonViaZarrsCodecs, replace(config, chunk_size=chunk_size))
        for config in _ZARR_CONFIGS
        if not config.only_python_zarr
    ]
    + [(AvailableFormats.OM, replace(config, chunk_size=chunk_size)) for config in _OM_CONFIGS]
    for chunk_size in CHUNKS.values()
}

REGISTERED_FORMAT_INVENTORY: Dict[tuple[int, int, int], List[Tuple[AvailableFormats, str]]] = {
    chunk_size: [(format, register_config(config)) for format, config in config_list]
    for chunk_size, config_list in CONFIGURATION_INVENTORY.items()
}

# register all configurations
all_hashes = [
    register_config(config)
    for format_config_list in CONFIGURATION_INVENTORY.values()
    for format_configuration, config in format_config_list
    if not format_configuration == AvailableFormats.ZarrTensorStore
    and not format_configuration == AvailableFormats.ZarrPythonViaZarrsCodecs
]
assert len(all_hashes) == len(set(all_hashes)), "Duplicate configuration hashes found!"
