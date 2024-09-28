"""
Some functionality to write and read .wav files
"""

import io
import numpy as np

from typing import Any

HEADER = b"RIFF"
FORMAT_HEADER = b"WAVE"
FORMAT_CHUNK_HEADER = b"fmt "
FORMAT_DATA_LEN = 16
PCM_INT = 1
IEEE_FLOAT = 3
DATA_HEADER = b"data" 


class WavReader:
    def __init__(self, filepath: str) -> None:
        """
        :param filepath:
        :return:
        """
        self.filename = filepath
        if not self.filename.endswith(".wav"):
            raise ValueError(f"Filepath must end in .wav")

    @staticmethod
    def _check_start_header(data: bytes) -> None:
        """
        Checks that the file header is correct
        :param data: File data in bytes
        :return:
        """
        header = data[0:4]
        if header != HEADER:
            raise RuntimeError(f"Invalid header {header.decode()}")

    @staticmethod
    def _check_filetype_header(data: bytes) -> None:
        """
        :param data:
        :return:
        """
        ft_header = data[8:12]
        if ft_header != FORMAT_HEADER:
            raise RuntimeError(f"Invalid file type header {ft_header.decode()}")

    @staticmethod
    def _check_format_marker(data: bytes) -> None:
        """
        :param data:
        :return:
        """
        form_chunk = data[12:16]
        if form_chunk != FORMAT_CHUNK_HEADER:
            raise RuntimeError(f"Invalid format chunk {form_chunk.decode()}")

    @staticmethod
    def _check_format_data_len(data: bytes) -> None:
        """
        :param data:
        :return:
        """
        format_data_len = int.from_bytes(data[16:20], "little")
        if format_data_len != FORMAT_DATA_LEN:
            raise RuntimeError(f"Invalid format data length {f_data_len}")

    @staticmethod
    def _find_data_block_header_end(data: bytes, start: int) -> None:
        """
        :param data:
        :param start:
        :return:
        """
        LIMIT = 1000
        while start < LIMIT:
            data_block = data[start:start + 4]
            if data_block == DATA_HEADER:
                return start + 4
            start += 1
        raise RuntimeError(f"Data block header not found")

    @staticmethod
    def _read_file_size(data: bytes) -> int:
        """
        :param data:
        :return:
        """
        return int.from_bytes(data[4:8], "little") + 8
    
    @staticmethod
    def _read_channels(data: bytes) -> int:
        """
        :param data:
        :return:
        """
        channels =  int.from_bytes(data[22:24], "little")
        if channels not in [1, 2]:
            raise RuntimeError(f"Invalid amount of channels {channels}")
        return channels

    @staticmethod
    def _read_audio_format(data: bytes) -> int:
        """
        :param data:
        :return:
        """
        format_type = int.from_bytes(data[20:22], "little")
        if format_type not in [PCM_INT, IEEE_FLOAT]:
            raise RuntimeError(f"Unknown audio format {format_type}")
        return format_type

    @staticmethod
    def _read_sample_rate(data: bytes) -> int:
        """
        :param data:
        :return:
        """
        return int.from_bytes(data[24:28], "little")

    @staticmethod
    def _read_bytes_per_bloc(data: bytes) -> int:
        """
        :param data:
        :return:
        """
        return int.from_bytes(data[32:34], "little")
    
    @staticmethod
    def _read_bits_per_sample(data: bytes) -> int:
        """
        :param data:
        :return:
        """
        return int.from_bytes(data[34:36], "little")
 
    @staticmethod
    def _read_data_block_size(data: bytes, pos: int) -> tuple[int, int]:
        """
        :param data:
        :return:
        """
        block_size = int.from_bytes(data[pos:pos + 4], "little")
        return block_size, pos + 4

    def read(self) -> tuple[np.ndarray, int]:
        """
        Reads the wav file and returns a tuple containing the audio data as a numpy
        array and the sample rate
        :return:
        """
        with open(self.filename, "rb") as f:
            # Read the whole thing in one go
            data = f.read()
        # Do some validation
        self._check_start_header(data=data)
        self._check_filetype_header(data=data)
        self._check_format_marker(data=data)
        self._check_format_data_len(data=data)
        data_size_pos = self._find_data_block_header_end(data=data, start=36)
        # Read some stuff
        filesize = self._read_file_size(data=data)
        audio_format = self._read_audio_format(data=data)
        channels = self._read_channels(data=data)
        fs = self._read_sample_rate(data=data)
        bytes_per_bloc = self._read_bytes_per_bloc(data=data)
        bits_per_sample = self._read_bits_per_sample(data=data)
        data_size, data_start = self._read_data_block_size(data=data, pos=data_size_pos)
        print(f"[INFO] Wav file size {filesize} bytes, data size {data_size} bytes, "
              f"sample rate {fs} Hz, {channels} audio channel(s), {bits_per_sample} "
              f"bits per sample, {bytes_per_bloc} bytes per bloc, audio format "
              f"{audio_format}")
        samples = int(data_size / (bits_per_sample / 8))
        if audio_format == PCM_INT:
            if bits_per_sample == 8:
                dtype = np.int8
            elif bits_per_sample == 16:
                dtype = np.int16
            else:
                # TODO: I guess at least 24 bit ints are also supported by wav
                raise RuntimeError(f"Integer data with size > 16 bits ({bits_per_sample})")
        # The validity of the format is already checked in _read_audio_format()
        else:
            if bits_per_sample == 32:
                dtype = np.float32
            else:
                # TODO: Perhaps other sizes should be allowed too
                raise RuntimeError(f"Float format with size != 32 ({bits_per_sample})")
        
        buffer = np.zeros(shape=(samples, ), dtype=dtype)
        cursor = data_start  # Start of the data section
        for i in range(samples):
            block = data[cursor:cursor + bytes_per_bloc]
            if audio_format == PCM_INT:
                num = int.from_bytes(block, "little")
            else:
                num = float.from_bytes(block, "little")
            buffer[i] = num
            cursor += bytes_per_bloc

        return buffer.reshape((-1, channels)), fs


class WavWriter:
    def __init__(self, file_obj: io.BufferedWriter, channels: int, bits_per_sample: int,
                 sample_rate: int) -> None:
        """
        :param file_obj: File handle to a opened file
        :param channels: Amount of audio channels
        :param bits_per_sample: AKA sample width
        :param sample_rate: Sampling rate in Hz
        :return:
        """
        self.file_obj = file_obj
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.fs = sample_rate
        self._total_size = 0
        self._data_size = 0
        self._write_num_channels()
        self._write_sample_rate()
        self._write_bits_per_sample()
        self._write_bytes_per_sec()
        self._write_bytes_per_block()
        self._init_header()
        self._total_size += 44

    def _init_header(self) -> None:
        """
        Writes parts of the wav header but "dynamic" things like the file size will
        need to be written/updated later
        :return:
        """
        # Assure that we are at the start of the file
        if self.file_obj.tell() != 0:
            self.file_obj.seek(0)
        self.file_obj.write(HEADER)
        self.file_obj.seek(8)
        self.file_obj.write(FORMAT_HEADER)
        self.file_obj.write(FORMAT_CHUNK_HEADER)
        self.file_obj.write(FORMAT_DATA_LEN.to_bytes(4, "little"))
        self.file_obj.seek(36)
        self.file_obj.write(DATA_HEADER)

    def _write_file_size(self) -> None:
        """
        :return:
        """
        self.file_obj.seek(4)
        self.file_obj.write(self._total_size.to_bytes(4, "little"))

    def _write_audio_format(self, form: int) -> None:
        """
        :param form:
        :return:
        """
        if form not in [PCM_INT, IEEE_FLOAT]:
            raise RuntimeError(f"Unknown audio format {form}")
        self.file_obj.seek(20)
        self.file_obj.write(form.to_bytes(2, "little"))

    def _write_num_channels(self) -> None:
        """
        :return:
        """
        self.file_obj.seek(22)
        self.file_obj.write(self.channels.to_bytes(2, "little"))

    def _write_sample_rate(self) -> None:
        """
        :return:
        """
        self.file_obj.seek(24)
        self.file_obj.write(self.fs.to_bytes(4, "little"))

    def _write_bytes_per_sec(self) -> None:
        """
        :return:
        """
        a = (self.fs * self.bits_per_sample * self.channels) / 8
        # I suppose the decimal part should be 0
        if not a.is_integer():
            raise RuntimeError(f"Bytes per sec % 8 != 0: {a}")
        self.file_obj.seek(28)
        self.file_obj.write(int(a).to_bytes(4, "little"))

    def _write_bytes_per_block(self) -> None:
        """
        :return:
        """
        a = (self.bits_per_sample * self.channels) / 8
        # I suppose the decimal part should be 0
        if not a.is_integer():
            raise RuntimeError(f"Bytes per block % 8 != 0: {a}")
        self.file_obj.seek(32)
        self.file_obj.write(int(a).to_bytes(2, "little"))

    def _write_bits_per_sample(self) -> None:
        """
        :return:
        """
        self.file_obj.seek(34)
        self.file_obj.write(self.bits_per_sample.to_bytes(2, "little"))

    def _write_data_size(self) -> None:
        """
        :return:
        """
        self.file_obj.seek(40)
        self.file_obj.write(self._data_size.to_bytes(4, "little"))

    def write_data(self, indata: np.ndarray, frames: int, time: int, status: Any) -> None:
        """
        :param indata:
        :param frames:
        :param time:
        :param status:
        :return:
        """
        # Mark these as unused for now
        _ = frames
        _ = time
        _ = status
        dtype = indata.dtype
        # TODO: Perhaps we don't need to check the datatype every time this function
        # is called
        if dtype == np.int8:
            self._write_audio_format(form=PCM_INT)
        elif dtype == np.int16:
            self._write_audio_format(form=PCM_INT)
        elif dtype == np.float32:
            self._write_audio_format(form=IEEE_FLOAT)
        else:
            raise RuntimeError(f"Unknown data type / audio format {dtype}")
        new_bytes = indata.tobytes()
        n = len(new_bytes)
        self.file_obj.seek(0, io.SEEK_END)
        self.file_obj.write(new_bytes)
        self._data_size += n
        self._total_size += n
        self._write_file_size()
        self._write_data_size()

    def get_written_bytes(self) -> int:
        """
        Returns the amount of bytes currently written to the file
        :return:
        """
        return self._total_size

