#!/home/henu/brogramming/audio_stuff/.venv/bin/python3.12
"""
A humble effort to create some sort of a spectrogram for audio
"""

import wave
import argparse
import numpy as np
import matplotlib.pyplot as plt


def wave_read(fname: str) -> tuple[np.ndarray, int]:
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


def spectrogram(data: np.ndarray, fs: int, bins: int = 100) -> None:
    """
    :param data:
    :param fs:
    :param bins:
    :return:
    """
    if len(data.shape) >= 2:
        raise ValueError("Only one channel is supported")
    n = data.shape[0]
    dt = n // bins  # Amount of samples in one bin
    fig = plt.figure()
    s, e = 0, dt  # Start and end indices
    t = np.zeros(shape=(bins, ))
    freqs = np.arange(0, dt // 2 + 1, dtype=int) * (fs / dt)
    amps = np.zeros(shape=(t.shape[0], freqs.shape[0]))
    for i in range(bins):
        t[i] = (e + s) / 2 / fs
        amps[i] = np.abs(np.fft.rfft(data[s:e])) / fs
        s = e
        e += dt
    amps = amps.T
    plt.pcolormesh(t, freqs, amps, shading="gouraud")
    c = plt.colorbar()
    c.set_label("Amplitude [dB/Hz]")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(prog="spegram.py",
                                     description="Plots a spectrogram of the given .wav file")
    parser.add_argument("filename", help="Name of the .wav file")
    args = parser.parse_args()
    input_path = args.filename
    data, fs = wave_read(fname=input_path)
    spectrogram(data=data[:, 0], fs=fs, bins=800)


if __name__ == "__main__":
    main()

