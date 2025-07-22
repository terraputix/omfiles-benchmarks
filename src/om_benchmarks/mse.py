import json
from pathlib import Path
from typing import Dict

import numpy as np


def mean_squared_error(original: np.ndarray, decompressed: np.ndarray) -> float:
    return float(np.mean((original - decompressed) ** 2))


class MSECache:
    def __init__(self, path: Path):
        self.path = path
        self._cache: Dict[str, float] = self._load()

    def _load(self) -> Dict[str, float]:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._cache, f)

    def set(self, file_path: str, mse: float) -> None:
        self._cache[file_path] = mse
        self.save()

    def get(self, file_path: str) -> float:
        if file_path not in self._cache:
            raise KeyError(f"MSE value for '{file_path}' not found in cache.")
        return self._cache[file_path]
