from typing import Literal, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.signal.windows import tukey


class PSDCalculator:
    """
    Calculates the Power Spectral Density (PSD) for each channel.
    Wrapper around scipy.signal.welch.
    """

    def __init__(
        self,
        fs: float,
        normalize: bool = False,
        welch_nperseg: Optional[int] = None,
        welch_window="hann",
        welch_noverlap: Optional[int] = None,
        welch_nfft: Optional[int] = None,
        welch_detrend: str = "constant",
        welch_return_onesided: bool = True,
        welch_scaling: str = "density",
        welch_axis: int = -1,
        welch_average: str = "mean",
        fft_taper_alpha_value: float = 0.25,
        fft_win_len_sec: float = 4,
        fft_fmin: Optional[float] = None,
        fft_fmax: Optional[float] = None,
        mode: Literal["welch", "fft"] = "welch",
    ):
        """
        Initialize the PSDCalculator with the necessary parameters.

        Args:
            mode (Literal["fft","welch"]): mode for computing psd in PSDCalculatora. Defaults to "welch".
            fs (float): Sampling frequency.
            normalize (bool): If True, applies 1/f normalization.
            welch_nperseg (int, optional): Length of each segment for Welch's method.
            welch_window (str or tuple or array): Desired window to use.
            welch_noverlap (int, optional): Number of points to overlap between segments.
            welch_nfft (int, optional): Number of FFT points.
            welch_detrend (str): Specifies how to detrend each segment.
            welch_return_onesided (bool): If True, return one-sided PSD.
            welch_scaling (str): Selects between 'density' and 'spectrum'.
            welch_axis (int): Axis over which to compute the PSD.
            welch_average (str): Method to average the PSD.
            taper_alpha_value (float, optional): Shape parameter of the Tukey window, r
                epresenting the fraction of the window inside the cosine tapered region.
                If zero, the Tukey window is equivalent to a rectangular window.
                If one, the Tukey window is equivalent to a Hann window. Defaults to 0.25.

        """

        self.mode = mode
        self.oof_scale = normalize
        self.fs = fs
        self.welch_nperseg = welch_nperseg or self.fs * 2
        self.welch_window = welch_window
        self.welch_noverlap = welch_noverlap
        self.welch_nfft = welch_nfft
        self.welch_detrend = welch_detrend
        self.wlech_return_onesided = welch_return_onesided
        self.welch_scaling = welch_scaling
        self.welch_axis = welch_axis
        self.welch_average = welch_average
        self.freqs = None
        # parameters for fft [TODO not implemented yet]
        self.fft_taper_alpha_value = fft_taper_alpha_value
        self.fft_win_len_sec = fft_win_len_sec
        self.fft_fmin = fft_fmin
        self.fft_fmax = fft_fmax
        if self.mode == "fft":
            self.calc_settings()

    def __repr__(self) -> str:
        welch_param = f"""
        welch_nperseg: {self.welch_nperseg}
        welch_window: {self.welch_window}
        welch_noverlap: {self.welch_noverlap}
        welch_nfft: {self.welch_nfft}
        welch_detrend: {self.welch_detrend}
        wlech_return_onesided: {self.wlech_return_onesided}
        welch_scaling: {self.welch_scaling}
        welch_axis: {self.welch_axis}
        welch_average: {self.welch_average}
        """
        fft_param = f"""
        fft_taper_alpha_value: {self.fft_taper_alpha_value}
        fft_win_len_sec: {self.fft_win_len_sec}
        fft_fmin: {self.fft_fmin}
        fft_fmax: {self.fft_fmax}
        """
        if self.mode == "fft":
            params = fft_param
        elif self.mode == "welch":
            params = welch_param
        return (
            f"""PSDCalculator using Welch with params:
            mode: {self.mode}
            normalization oof_scale: {self.oof_scale}
            fs: {self.fs}"""
            + params
        )

    def calc_settings(self):

        self.n_samples = int(np.round(self.fs * self.fft_win_len_sec))
        if self.n_samples % 2:  # N must be even
            self.n_samples += 1
        # define taper if nonzero alpha
        if self.fft_taper_alpha_value > 0:
            self.taper = tukey(self.n_samples, alpha=self.fft_taper_alpha_value)
            self.taperlen = len(self.taper)
        else:
            self.taperlen = 0

        self.df = self.fs / self.n_samples
        self.freqs_all = np.array(
            [
                (
                    self.df * n
                    if n < self.n_samples / 2
                    else self.df * (n - self.n_samples)
                )
                for n in range(self.n_samples)
            ]
        )
        self.fft_fmin = self.fft_fmin or np.min(np.abs(self.freqs_all))
        self.fft_fmax = self.fft_fmax or np.max(self.freqs_all)
        # self.keepind = np.greater_equal(self.freqs_all, 0)
        # self.freqs = self.freqs_all[self.keepind]
        self.keepind = np.logical_and(
            self.freqs_all >= self.fft_fmin, self.freqs_all <= self.fft_fmax
        )
        self.freqs = self.freqs_all[self.keepind]

    def dofft(self, arr):
        if self.fft_taper_alpha_value > 0:  # applying Tukey taper if necessary
            if arr.shape[0] != self.taperlen:
                self.taper = tukey(arr.shape[0], alpha=self.fft_taper_alpha_value)
            arr = arr * self.taper

        # conducting fft, calculating PSD
        spectra = np.abs(np.fft.fft(arr) ** 2) / self.df  # PSD = |X(f)^2| / df
        spectra[np.isinf(spectra)] = (
            1.0e-8  # replacing negative inf values (spectra power=0)
        )

        # limiting data to positive/real frequencies only (and convert to dB)
        spectra = np.log10(spectra.ravel()[self.keepind])

        return spectra

    def compute_psd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Power Spectral Density (PSD) for each channel.

        Args:
            data (np.ndarray): Input data
                Data always arrive in shape [time, channel] if axis==0

                NOT RECCOMENDED: You can play with parameter axis considering that
                            if axis==-1 you expect data shape [channels, time]
                            if axis==-0 you expect data shape [time, channels]

        Returns:
            freqs (np.ndarray): Frequency values corresponding to the PSD.
            psd (np.ndarray): Computed PSD of each channel.
        """
        if self.mode == "fft":
            psd = np.zeros((len(self.freqs), data.shape[1]))
            for ch_ix, channel_time_series in enumerate(data.T):
                psd[:, ch_ix] = self.dofft(channel_time_series)
        elif self.mode == "welch":
            # psd is returned as coeff x channel
            # self.freqs will have len == psd.shape[0]
            self.freqs, psd = welch(
                data,
                fs=self.fs,
                nperseg=self.welch_nperseg,
                window=self.welch_window,
                noverlap=self.welch_noverlap,
                nfft=self.welch_nfft,
                detrend=self.welch_detrend,
                return_onesided=self.wlech_return_onesided,
                scaling=self.welch_scaling,
                axis=self.welch_axis,
                average=self.welch_average,
            )
        else:
            raise ValueError(
                f"MODE {self.mode} not recognized should be or FFT or WELCH"
            )
        if self.oof_scale:
            psd = self.oof_normalization(psd, self.freqs)
        return self.freqs, psd

    @staticmethod
    def oof_normalization(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """
        Apply 1/f normalization to the PSD.

        Args:
            psd (np.ndarray): Power Spectral Density values.
            freqs (np.ndarray): Corresponding frequency values.

        Returns:
            np.ndarray: Normalized PSD.
        """
        # Avoid division by zero by setting 0 Hz to 1
        freqs_for_scaling = np.where(freqs == 0, 1, freqs)
        if psd.ndim > 1:
            freqs_for_scaling = np.tile(
                np.expand_dims(freqs_for_scaling, 1), (1, psd.shape[-1])
            )

        psd *= freqs_for_scaling  # Apply 1/f scaling
        return psd

    def extract_band(
        self, psd_data: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]
    ) -> np.ndarray:
        """
        Extracts the power from a specific frequency band.

        Args:
            psd_data (np.ndarray): The PSD data [freqs x channels].
            freqs (np.ndarray): Corresponding frequency values. Shape == psd_data.shape[0]
            band (tuple): Frequency band as (low, high).

        Returns:
            np.ndarray: Power in the specified frequency band for each channel.
        """
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        band_power = np.mean(psd_data[band_mask, :], axis=0)
        return band_power
