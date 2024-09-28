"""
Tool for recording audio (in .wav format) or some shit
"""

import time
import argparse
import numpy as np
import sounddevice as sd

from typing import Any
from wavfile_io import WavWriter

BITS_PER_SAMPLE = 16


def _format_nbytes(nbytes: int) -> str:
    """
    Formats a bit the number of bytes depending on its size
    :param nbytes:
    """
    mb = 1024 * 1024
    if nbytes < 1024:
        return f"{nbytes} B"
    elif 1024 <= nbytes < mb:
        return f"{nbytes / 1024:.2f} kB"
    else:
        return f"{nbytes / mb:.2f} MB"


def _flush_and_print(msg: str) -> None:
    """
    Clears the line and prints msg on it (so that no new line is printed)
    :param msg:
    :return:
    """
    print("\r\033[K", end="")
    print(f"\r{msg}", end="")


def record(out_file_path: str, fs: int, channels: int) -> None:
    """
    :param out_file_path:
    :param fs:
    :param channels:
    """
    if not out_file_path.endswith(".wav"):
        raise ValueError("File path must be of a .wav file")
    print("INFO: Starting recording, press CTRL + C (keyboard interrupt) to stop")
    try:
        duration = 0
        dt = .1
        with open(out_file_path, "wb") as f:
            fw = WavWriter(file_obj=f, channels=channels, bits_per_sample=BITS_PER_SAMPLE,
                           sample_rate=fs)
            with sd.InputStream(samplerate=fs, channels=channels, callback=fw.write_data,
                                dtype=f"int{BITS_PER_SAMPLE}"):
                while True:
                    duration += dt
                    nbytes = _format_nbytes(nbytes=fw.get_written_bytes())
                    msg = f"INFO: Duration: {duration:.1f} s, filesize: {nbytes} "
                    _flush_and_print(msg=msg)
                    time.sleep(dt) # Add some downtime

    except KeyboardInterrupt:
        print("\nINFO: Recording stopped")

    print(f"INFO: {out_file_path} created")


def main() -> None:
    fs_def = 44100
    channels_def = 1
    parser = argparse.ArgumentParser(prog="recorder.py",
                                     description="Simple audio recorder")
    parser.add_argument("filepath", help="Path of the output file")
    parser.add_argument("-f", "--samplerate", type=int, default=fs_def,
                        help="Sample rate (frame rate) used for recording [1/s]")
    parser.add_argument("-c", "--channels", type=int, default=channels_def,
                        help="Amount of audio channels - e.g. 1 for mono, 2 for stereo")
    args = parser.parse_args()
    out_path = args.filepath
    fs = args.samplerate
    channels = args.channels
    print(f"INFO: File name: {out_path}, fs: {fs}, channels: {channels}")
    record(out_file_path=out_path, fs=fs, channels=channels)


if __name__ == "__main__":
    main()

