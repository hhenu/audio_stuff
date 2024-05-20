#!/home/henu/brogramming/audio_stuff/.venv/bin/python3.12
"""
Some sort of Buffer class to hold the most recent audio data or some shit
"""

import io
import wave
import numpy as np

from typing import Any


class FileWriter:
    def __init__(self, file: wave.Wave_write) -> None:
        """
        :param file: File handler of a file to write stuff in
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
        data_b = indata.tobytes()
        self._size += len(data_b)
        self.file.writeframes(data_b)

    def get_written_bytes(self) -> int:
        """
        Returns the amount of bytes currently written to the file
        :return:
        """
        return self._size

