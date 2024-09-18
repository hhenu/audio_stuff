"""
Some functionality to write and read .wav files
"""

import io
import numpy as np

from typing import Any


class WavReader:
    def __init__(self, filepath: str) -> None:
        """
        :param filepath:
        :return:
        """
        self.filename = filepath


class WavWriter:
    def __init__(self, filepath: str) -> None:
        """
        :param filepath:
        :return:
        """
        self.file = file
        self._size = 0

    def write(self, indata: np.ndarray, frames: int, time: int, status: Any) -> None:
        """
        :param indata:
        :param frames:
        :param time:
        :param status:
        :return:
        """
        ...

    def get_written_bytes(self) -> int:
        """
        Returns the amount of bytes currently written to the file
        :return:
        """
        return self._size

