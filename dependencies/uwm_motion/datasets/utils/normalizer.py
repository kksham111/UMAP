from typing import Union
import numpy as np


class LinearNormalizer:
    """
    A class to linearly normalizes data to the range [-1, 1].
    """

    def __init__(self, scale: np.ndarray, offset: np.ndarray):
        """
        Initializes a new instance of the LinearNormalizer class with given statistics.

        Args:
            scale: The scale factor for normalization.
            offset: The offset for normalization.
        """
        self.scale = scale
        self.offset = offset

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes the input data using the stored statistics.
        """
        return (x - self.offset) / self.scale

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        Reconstructs the original data from the normalized data.
        """
        return x * self.scale + self.offset


class NestedDictLinearNormalizer(dict):
    """
    A class that applies linear normalization to values in a nested dictionary structure.
    """

    def __init__(self, stats: dict[str, Union[tuple[np.ndarray, np.ndarray], dict]]):
        """
        Initializes a new instance of the NestedDictLinearNormalizer class with given statistics.

        Args:
            stats: A dictionary containing statistics for each key. The values can either
                be tuples representing scale and offset values or dictionaries that require
                recursive scaling.
        """
        super().__init__()
        for k, v in stats.items():
            if isinstance(v, dict):
                self[k] = NestedDictLinearNormalizer(v)
            else:
                self[k] = LinearNormalizer(np.array(v[0]), np.array(v[1]))

    def __call__(self, x: dict) -> dict:
        """
        Normalizes all values in the input dictionary based on the stored normalizers.
        """
        return {k: self[k](v) if k in self.keys() else v for k, v in x.items()}

    def reconstruct(self, x: dict) -> dict:
        """
        Reconstructs the original values from normalized values in the input dictionary.
        """
        return {
            k: self[k].reconstruct(v) if k in self.keys() else v for k, v in x.items()
        }
