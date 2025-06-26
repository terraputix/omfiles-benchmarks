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
        elif item:  # Skip empty strings from trailing commas
            result.append(int(item))
    return tuple(result)
