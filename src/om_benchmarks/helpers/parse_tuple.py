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
