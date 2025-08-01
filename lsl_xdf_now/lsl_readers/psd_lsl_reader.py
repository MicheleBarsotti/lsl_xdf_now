"""
Class for reading an LSL stream and applying PSD
"""

from typing import Dict, List, Tuple

import numpy as np

from ..data_processing.buffer import RingBuffer
from ..data_processing.filters import PreprocessIIR
from ..data_processing.psd_calculator import PSDCalculator
from .lsl_iterator import LSLIterator


class PSDWithLSLreader:
    def __init__(
        self,
        fs,
        n_eeg_channels,
        stream_name_eeg: str = "",
        stream_type_eeg: str = "EEG",
        bands_cutoff: List[Tuple[float, float]] = [],
        bands_names: List[str] = [],
        buffer_size_seconds=4,
        preproc_low=1,
        preproc_high=0,  # no HP filter
        preproc_notch_filter=50,
        normalize_oof: bool = True,
        **psd_kwargs,
    ) -> None:
        self.bands_cutoff = bands_cutoff
        self.bands_names = bands_names
        ## the buferr of the lslBufferizer is not used
        self.n_eeg_channels = n_eeg_channels
        self.lsl_bufferizer = LSLIterator(
            n_channels=n_eeg_channels,
            stream_type=stream_type_eeg,
            stream_name=stream_name_eeg,
        )
        self.channel_in_lsl = self.lsl_bufferizer._count_channel
        ## real buffer implemented here
        self.buffer_size = int(fs * buffer_size_seconds)
        self.buffer = RingBuffer(time_len=self.buffer_size, channels=n_eeg_channels)
        if n_eeg_channels > self.channel_in_lsl:
            raise ValueError(
                f"Requested {n_eeg_channels} channels, but only {self.channel_in_lsl} are available in the LSL stream."
            )
        ## filter for removing slow freq and notch filter
        if preproc_high == 0:
            preproc_high = fs / 2
        if preproc_low >= preproc_high:
            raise ValueError(
                f"Low cut frequency {preproc_low} must be lower than high cut frequency {preproc_high}."
            )
        if preproc_notch_filter >= fs / 2:
            raise ValueError(
                f"Notch filter frequency {preproc_notch_filter} must be lower than Nyquist frequency {fs / 2}."
            )
        if preproc_notch_filter > 0 and preproc_notch_filter < preproc_low:
            raise ValueError(
                f"Notch filter frequency {preproc_notch_filter} must be higher than low cut frequency {preproc_low}."
            )
        if preproc_notch_filter > 0 and preproc_notch_filter > preproc_high:
            raise ValueError(
                f"Notch filter frequency {preproc_notch_filter} must be lower than high cut frequency {preproc_high}."
            )

        self.filter_eeg = PreprocessIIR(
            low=preproc_low,
            high=preproc_high,
            sfreq=fs,
            notch_filter=preproc_notch_filter,
            num_channels=n_eeg_channels,
        )
        self.psd_computer = PSDCalculator(fs=fs, normalize=normalize_oof, **psd_kwargs)
        self._comunicate_buffer_ready = True

    def process(self):
        while True:
            chunks, ts = self.lsl_bufferizer.inlet.pull_chunk()
            if not ts:
                continue
            data = {
                "eeg": np.concatenate(chunks).reshape(len(chunks), self.channel_in_lsl)[
                    :, : self.buffer._channels
                ]
            }
            data_f = self.filter_eeg(data)
            if data_f:
                self.buffer.extend(data_f["eeg"].T)
            if not self.buffer.ready:
                self.log(
                    f"Buffer not ready. {self.buffer.percentage_ready()}", end="\r"
                )
                continue
            if self._comunicate_buffer_ready:
                self.log("BUFFER READY TO GO!!!")
                self._comunicate_buffer_ready = False
            psd_array = np.empty(0)
            for i in range(self.buffer._channels):
                channel_data = self.buffer.values[i, :]
                freqs, psd = self.psd_computer.compute_psd(channel_data)
                if psd_array.size == 0:
                    psd_array = psd
                else:
                    psd_array = np.c_[psd_array, psd]
            # psd_array is frequency x channels (e.g. 251 x 8)
            return freqs, psd_array

    def _extract_bands(
        self,
        psd_data: np.ndarray,
        freqs: np.ndarray,
        bands_cutoff: List[Tuple[float, float]],
        bands_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Extracts the power from a specific frequency band.

        Args:
            psd_data (np.ndarray): The PSD data [freqs x channels].
            freqs (np.ndarray): Corresponding frequency values. Shape == psd_data.shape[0]
            band (tuple): Frequency band as (low, high).

        Returns:
            np.ndarray: Power in the specified frequency band for each channel.
        """
        out_dict = {}
        for band_name, cutoff in zip(bands_names, bands_cutoff):
            out_dict[band_name] = self.psd_computer.extract_band(
                psd_data=psd_data, freqs=freqs, band=cutoff
            )
        return out_dict

    def __call__(self) -> Dict[str, np.ndarray]:
        freqs, psd = self.process()
        return self._extract_bands(
            psd_data=psd,
            freqs=freqs,
            bands_cutoff=self.bands_cutoff,
            bands_names=self.bands_names,
        )

    def log(self, msg: str, **kwargs):
        print(msg, **kwargs)


if __name__ == "__main__":
    ps = PSDWithLSLreader(
        fs=250,
        n_eeg_channels=8,
    )

    for _ in range(3):
        ff, pp = ps.process()
