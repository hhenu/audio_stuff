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


def spectrogram(data: np.ndarray, fs: int, nperseg: int, noverlap: int) -> None:
    """
    Computes and plots a spectrogram  of the given data
    :param data: Some data that changes as a function of time
    :param fs: Sample rate
    :param nperseg: Amount of data points to use for one sequence of data
    :param noverlap: Amount of points that overlap with the previous sequence
    """
    if len(data.shape) >= 2:
        raise ValueError("Only one channel is supported")
    if noverlap >= nperseg:
        raise ValueError("noverlap can not be larger than nperseg")
    n = data.shape[0]
    step = nperseg - noverlap
    nsegs = n // step
    t = np.zeros(shape=(nsegs, ), dtype=float)
    freqs = np.arange(0, nperseg // 2 + 1, dtype=int) * (fs / nperseg)
    amps = np.zeros(shape=(t.shape[0], freqs.shape[0]), dtype=float)
    s, e = 0, nperseg  # Start and end indices
    for i in range(nsegs):
        t[i] = (e + s) / 2 / fs
        amp =  np.abs(np.fft.rfft(data[s:e])) / fs
        if amp.shape[0] < amps.shape[1]:
            # Pad the output of fft with some zeros if it's too small
            pad = np.zeros(shape=(amps.shape[1] - amp.shape[0], ))
            amp = np.append(amp, pad)
        amps[i] = amp
        s += step
        e += step
    amps = amps.T
    fig = plt.figure()
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
    spectrogram(data=data[:, 0], fs=fs, nperseg=200, noverlap=50)


if __name__ == "__main__":
    main()

