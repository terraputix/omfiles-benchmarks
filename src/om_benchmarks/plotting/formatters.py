def format_bytes(x: float, pos: int) -> str:
    """Format bytes into human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if x < 1024.0:
            return f"{x:.1f}\\,{unit}"
        x /= 1024.0
    return f"{x:.1f}\\,TB"


def format_time(x: float, pos: int) -> str:
    """Format time into human readable format"""
    if x < 1:
        return f"{x * 1000:.1f}\\,ms"
    else:
        return f"{x:.2f}\\,s"
