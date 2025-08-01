import hashlib
import pickle
from dataclasses import replace
from typing import Dict, List, Tuple, cast

import hdf5plugin
import numcodecs
import numcodecs.zarr3
import omfiles.numcodecs
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

_BASELINE_CONFIG = BaselineConfig(chunk_size=CHUNKS["small"])
_NETCDF_BEST = NetCDFConfig(chunk_size=CHUNKS["small"], compression="zlib", compression_level=3)
_HDF5_BEST = HDF5Config(chunk_size=CHUNKS["small"], compression=HBlosc(cname="lz4", clevel=4, shuffle=HBlosc.SHUFFLE))
_ZARR_BEST = ZarrConfig(chunk_size=CHUNKS["small"], compressor=NBlosc(cname="lz4", clevel=4, shuffle=NBlosc.BITSHUFFLE))
_OM_BEST = OMConfig(chunk_size=CHUNKS["small"], compression="pfor_delta_2d", scale_factor=20, add_offset=0)

_NETCDF_CONFIGS = [
    NetCDFConfig(chunk_size=CHUNKS["small"], compression=None),  # netcdf baseline: no compression
    NetCDFConfig(chunk_size=CHUNKS["small"], compression="szip", significant_digits=1),
    NetCDFConfig(chunk_size=CHUNKS["small"], compression="szip", significant_digits=2),
    NetCDFConfig(chunk_size=CHUNKS["small"], compression="szip", scale_factor=1.0),
    _NETCDF_BEST,
]

_HDF5_CONFIGS = [
    HDF5Config(chunk_size=CHUNKS["small"]),  # hdf5 baseline: no compression
    _HDF5_BEST,
    # https://hdfgroup.github.io/hdf5/develop/group___s_z_i_p.html#ga688fde8106225adf9e6ccd2a168dec74
    # https://hdfgroup.github.io/hdf5/develop/_h5_d__u_g.html#title6
    # 1st 'nn' stands for: H5_SZIP_NN_OPTION_MASK
    # 2nd 32 stands for: 32 pixels per block
    HDF5Config(chunk_size=CHUNKS["small"], compression="szip", compression_opts=("nn", 32), scale_offset=2),
    # https://hdfgroup.github.io/hdf5/develop/group___s_z_i_p.html#ga688fde8106225adf9e6ccd2a168dec74
    # https://hdfgroup.github.io/hdf5/develop/_h5_d__u_g.html#title6
    # 1st 'nn' stands for: H5_SZIP_NN_OPTION_MASK
    # 2nd 32 stands for: 32 pixels per block
    HDF5Config(chunk_size=CHUNKS["small"], compression="szip", compression_opts=("nn", 8)),
    HDF5Config(chunk_size=CHUNKS["small"], compression=HBlosc(cname="zstd", clevel=9, shuffle=HBlosc.SHUFFLE)),
    HDF5Config(chunk_size=CHUNKS["small"], compression=hdf5plugin.SZ(absolute=0.01)),
]

_ZARR_CONFIGS = [
    ZarrConfig(chunk_size=CHUNKS["small"], compressor=None),  # zarr baseline: no compression
    ZarrConfig(chunk_size=CHUNKS["small"], compressor=numcodecs.Blosc()),
    _ZARR_BEST,
    ZarrConfig(chunk_size=CHUNKS["small"], compressor=NBlosc(cname="zstd", clevel=3, shuffle=NBlosc.BITSHUFFLE)),
    ZarrConfig(
        zarr_format=3,
        chunk_size=CHUNKS["small"],
        serializer=omfiles.zarr3.PforSerializer(),  # type: ignore
        filter=numcodecs.zarr3.FixedScaleOffset(offset=0, scale=20, dtype="f4", astype="i4"),
        only_python_zarr=True,
    ),
    ZarrConfig(
        zarr_format=3,
        chunk_size=CHUNKS["small"],
        serializer=numcodecs.zarr3.PCodec(mode_spec="auto"),
        filter=numcodecs.zarr3.FixedScaleOffset(offset=0, scale=20, dtype="f4", astype="i4"),
        only_python_zarr=True,
    ),
    ZarrConfig(
        zarr_format=3,
        chunk_size=CHUNKS["small"],
        serializer=numcodecs.zarr3.PCodec(level=8, mode_spec="auto"),
        filter=numcodecs.zarr3.FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="i4"),
        only_python_zarr=True,
    ),
    ZarrConfig(
        zarr_format=3,
        chunk_size=CHUNKS["small"],
        serializer=numcodecs.zarr3.PCodec(),
        only_python_zarr=True,
    ),
]

_OM_CONFIGS = [
    _OM_BEST,
    OMConfig(chunk_size=CHUNKS["small"], compression="pfor_delta_2d", scale_factor=100, add_offset=0),
    OMConfig(chunk_size=CHUNKS["small"], compression="fpx_xor_2d"),
]

CONFIGURATION_INVENTORY: Dict[tuple[int, int, int], List[Tuple[AvailableFormats, FormatWriterConfig]]] = {
    chunk_size: [
        (AvailableFormats.Baseline, replace(config, chunk_size=chunk_size))
        for config in [cast(FormatWriterConfig, _BASELINE_CONFIG)]
    ]
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
