from matplotlib.ticker import FuncFormatter


def format_bytes(x: float, pos: int) -> str:
    """Format bytes into human readable format using binary prefixes (KiB, MiB, GiB, TiB)"""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if x < 1024.0:
            return f"{x:.1f}\\,{unit}"
        x /= 1024.0
    return f"{x:.1f}\\,TiB"


def format_time(x: float, pos: int) -> str:
    """Format time into human readable format"""
    if x < 1e-3:
        return f"{x * 1e6:.1f}\\,$\mu \mathrm{{s}}$"
    elif x < 0.1:
        return f"{x * 1e3:.1f}\\,ms"
    else:
        return f"{x:.1f}\\,s"


TIME_FORMATTER = FuncFormatter(format_time)
BYTES_FORMATTER = FuncFormatter(format_bytes)
