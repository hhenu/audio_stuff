"""
Plays a single .wav file
"""

import wave
import argparse
import numpy as np
import sounddevice as sd


def _wave_read(fname: str) -> tuple[np.ndarray, int]:
    """
    Stolen from https://stackoverflow.com/questions/54174160/how-to-get-numpy-
    arrays-output-of-wav-file-format
    
    :param fname: Name of the .wav file
    :return: Tuple containing the audio data as an array and the sample rate
    """
    if not fname.endswith(".wav"):
        raise ValueError("File must be a .wav file")
    print(f"INFO: Reading file {fname}")
    with wave.open(fname, "rb") as f:
        buf = f.readframes(f.getnframes())
        temp = np.frombuffer(buf, dtype=f"int{f.getsampwidth() * 8}")
        return np.reshape(temp, (-1, f.getnchannels())), f.getframerate()


def play_audio(fname: str, speed: float) -> None:
    """
    Plays the audio using the given speed multiplier
    :param fname:
    :param speed:
    :return:
    """
    data, fs = _wave_read(fname=fname)
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

