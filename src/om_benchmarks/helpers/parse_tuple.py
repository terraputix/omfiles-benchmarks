import re

from omfiles.types import BasicSelection


def parse_tuple(string: str) -> BasicSelection:
    # Remove parentheses and split by comma
    items = string.strip("()").split(",")
    # Convert each item to the appropriate type
    result = []
    for item in items:
        item = item.strip()
        if item == "...":
            result.append(Ellipsis)
        elif item.lower() == "none":
            result.append(None)
        elif ".." in item:
            # Handle range notation (e.g., "190..200")
            range_parts = item.split("..")
            if len(range_parts) == 2:
                start = int(range_parts[0]) if range_parts[0].strip() else None
                end = int(range_parts[1]) if range_parts[1].strip() else None
                result.append(slice(start, end))
        elif item:  # Skip empty strings from trailing commas
            result.append(int(item))
    return tuple(result)


def pretty_read_index(read_index_str: str) -> str:
    """
    Converts a string like '(slice(100,104), slice(200,204,2), Ellipsis)'
    to '[100:104, 200:204:2, ...]'
    Handles step=None as [start:stop] (no step shown).
    """
    if not read_index_str:
        return ""
    s = read_index_str.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    s = s.replace("Ellipsis", "...")

    # Function to convert slice(start, stop, step) to start:stop:step
    def slice_repl(match):
        start, stop, step = match.groups()
        # Remove 'None' step or if step is missing
        if step is None or step == "None":
            return f"{start}:{stop}"
        else:
            return f"{start}:{stop}:{step}"

    # Replace slice(start, stop, step) or slice(start, stop)
    s = re.sub(r"slice\(\s*([^,]+)\s*,\s*([^,]+)\s*(?:,\s*([^)]+)\s*)?\)", slice_repl, s)
    # Remove any extra whitespace
    s = ", ".join(part.strip() for part in s.split(","))
    return f"[{s}]"
