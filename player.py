"""
Plays a single .wav file
"""

import argparse
import numpy as np
import sounddevice as sd

from wavfile_io import WavReader


def play_audio(fname: str, speed: float) -> None:
    """
    Plays the audio using the given speed multiplier
    :param fname:
    :param speed:
    :return:
    """
    data, fs = WavReader(filepath=fname).read()
    print(f"INFO: Playing {fname} with {speed}x speed")
    sd.play(data, samplerate=int(fs * speed))
    sd.wait()


def main() -> None:
    speed_def = 1.
    parser = argparse.ArgumentParser(prog="player.py",
                                    description="Plays some wave (.wav) files")
    parser.add_argument("filepath", help="Path of the input .wav file")
    parser.add_argument("-s", "--speed", type=float, default=speed_def,
                        help="Speed multiplier for playback, e.g. 1.5 for 1.5x speed")
    args = parser.parse_args()
    input_path = args.filepath
    speed = args.speed
    play_audio(fname=input_path, speed=speed)


if __name__ == "__main__":
    main()

