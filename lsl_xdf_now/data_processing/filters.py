from logging import warning
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, firwin, lfilter, lfiltic, sosfilt, sosfilt_zi

BANDS_CUTOFF = ((1.0, 3.0), (4.0, 7.0), (8.0, 13.0), (14.0, 30.0), (31.0, 40.0))
from typing import Sequence, Union


class DimensionError(Exception):
    """Raise when the dimension of signals not as expected"""

    pass


# @dataclass
class IIR:
    """
    IIR multi-channel filter
    """

    # num_channel: int
    # sampling_frequency: Union[int, float]
    # axis: int = field(repr=False, default=-1)
    # raw_enabled: bool = field(repr=False, default=False)
    # coeffs: list[tuple] = field(init=False, repr=False, default_factory=list)
    # past_zi: list[np.ndarray] = field(init=False, repr=False, default=None)

    def __init__(
        self,
        num_channel: int,
        sampling_frequency: Union[int, float],
        raw_enabled: bool = False,
    ) -> None:
        self.num_channel = num_channel
        self.sampling_frequency = sampling_frequency
        self.raw_enabled = raw_enabled
        self.coeffs = []
        self.past_zi = None

    def _init_zi(self, signals: Union[list, np.ndarray]) -> None:
        """
        Initialize initial condition for first sample to reduce transient-state time of signals
        :param signals: Two dimensional matrix of [samples x channels]
        :type signals: List or numpy array
        """
        self.past_zi = []
        first_sample = signals[..., 0]

        # Change dimension to match initial_zi dimension
        first_sample = np.expand_dims(np.expand_dims(first_sample, axis=-1), axis=0)

        for coeff in self.coeffs:
            initial_zi = sosfilt_zi(coeff)
            initial_zi = np.repeat(
                np.expand_dims(initial_zi, axis=1), self.num_channel, axis=1
            )
            initial_zi *= first_sample
            self.past_zi.append(initial_zi)

    def set_raw_enabled(self, state: bool) -> None:
        self.raw_enabled = state
        self.past_zi = None

    def add_filter(
        self, order: int, cutoff: Union[Sequence, int, float], filter_type: str
    ) -> None:
        """
        Add filter into cascading pipeline
        :param int order: An order of filter.
        :param Union[Sequence, int, float] cutoff: A critical frequency of the filter.
        :param str filter_type: Filter type can be 'lowpass', 'highpass', 'bandstop' and 'bandpass'.
        """

        new_filter_coeff = butter(
            order, cutoff, filter_type, output="sos", fs=self.sampling_frequency
        )

        self.coeffs.append(new_filter_coeff)

    def add_sos(self, sos: np.ndarray) -> None:
        """
        Add sos filter into cascading pipeline
        :param ndarray sos: A filter coefficient.
        """
        self.coeffs.append(sos)

    def __call__(self, arr) -> np.ndarray:
        return self.filter(raw_signal=arr)

    def filter(self, raw_signal: Union[list, np.ndarray]) -> np.ndarray:
        """
        Filter a sequence of multi-channel samples
        :param raw_signal: Two dimensional matrix of [samples x channels]
        :type raw_signal: Union[list, np.ndarray]
        :return np.ndarray
        """

        # Check if input is list or numpy array
        if isinstance(raw_signal, list):
            filt_signal = np.array(list(zip(*raw_signal)))
        elif isinstance(raw_signal, np.ndarray):
            filt_signal = raw_signal.T

        # If raw_mode then return
        if self.raw_enabled:
            return filt_signal.T

        # Check input correctness
        signal_dim = filt_signal.shape
        if len(signal_dim) != 2:
            raise DimensionError("Input signal dimension must be equal to 2")
        if signal_dim[0] != self.num_channel:
            raise DimensionError(
                f"Number of channels must be equal to {self.num_channel}. Passed array is of shape {signal_dim}"
            )

        if self.past_zi is None:
            self._init_zi(filt_signal)

        for index, (sos, past_zi) in enumerate(zip(self.coeffs, self.past_zi)):
            filt_signal, zi = sosfilt(sos, filt_signal, zi=past_zi)
            self.past_zi[index] = zi

        return filt_signal.T


class PreprocessIIR:
    def __init__(
        self,
        low_cut: float = 1.0,
        high_cut: float = 40.0,
        sfreq: float = 500.0,
        filter_order: int = 4,
        num_channels: int = 32,
        notch_filter: Optional[Union[int, float]] = 50,
        notch_filter_width: float = 5,
        notch_filter_order: int = 2,
    ):
        """
        Operates IIR filer on data coming in the form  \n

        Args:
            name (str): name of the node
            low (float, optional): low cut off to 1.0.
            high (float, optional): high cut off. Defaults to 40.0.
            sfreq (float, optional): sample frequency. Defaults to 500.0.
            filter_order (int, optional): filter order. Defaults to 4.
            num_channels (int, optional): how many channel has the data (rows). Defaults to 32.
            notch_filter_width (float): width of the Notch filter.
                Band stop filter will be in range [w0-width:w0+width].  Default 5.
            notch_filter_order (int): order of the Notch filter. Default 2..
        Raises:
            TypeError: if notch_filter is not one between Union[NotchFilter, float]
        """
        self.desc = ""
        self.iir_filter = IIR(num_channel=num_channels, sampling_frequency=sfreq)
        self.sfreq = sfreq
        self.num_channels = num_channels
        self._do_bp_filter = True
        cutoff: Union[Tuple[float, float], float, int, bool] = False
        btype = ""
        high_cut = high_cut or 0
        low_cut = low_cut or 0
        if low_cut > 0 and high_cut > 0:
            cutoff = (low_cut, high_cut)
            btype = "bandpass"
        elif low_cut > 0:
            cutoff = low_cut
            btype = "highpass"
        elif high_cut > 0:
            cutoff = high_cut
            btype = "lowpass"
        else:
            self._do_bp_filter = False

        if notch_filter is not None and notch_filter > 0:
            self.desc += f"| notch: 6th order bandstop {notch_filter-notch_filter_width}-{notch_filter+notch_filter_width} Hz |"
            self.iir_filter.add_filter(
                order=notch_filter_order,
                cutoff=(
                    notch_filter - notch_filter_width,
                    notch_filter + notch_filter_width,
                ),
                filter_type="bandstop",
            )

        else:
            self.notch_filter = None
        # create pipeline through composite function
        if self._do_bp_filter and cutoff and btype:
            self.iir_filter.add_filter(
                order=filter_order, cutoff=cutoff, filter_type=btype
            )

    def __call__(self, arr: NDArray) -> NDArray:
        if isinstance(arr, (int, float, np.number)):
            arr = np.array([arr])
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.shape[0] != self.iir_filter.num_channel:
            warning(
                f"""
              WRONG DIMENSION:
                arr.shape={arr.shape} whereas iir_filter.num_channel {self.iir_filter.num_channel}
                output is not accurate.
                Consider Reshaping the input before filtering or check dimensions
              """
            )
        return self.iir_filter.filter(arr.T).T


def get_filterbanked_eeg(
    eeg: np.ndarray,
    sample_frequency: Union[int, float],
    cut_offs: List[Tuple[float, float]],
    order: int = 3,
):
    """
    From eeg array (n_channels, n_samples)
    Args:
        eeg: (n_channels, n_samples) eeg ndarray
        sample_frequency: eeg sample rate
        cut_offs: list containing low cut and high cut for every filter bank band
        order: Butterworth bandpass filter order
    Returns
        The filter banked eeg (n_bands, n_channels, n_samples)
    """
    f_bank = ButterworthBandPassFilterBank(cut_offs, sample_frequency, order)
    return f_bank(eeg)


class LinearFilter:
    """
    Linear filter class
    """

    def __init__(self, b: np.ndarray, a: np.ndarray, num_channels: int, axis: int = -1):
        """

        :param b: The 'b' coefficients of the filter
        :param a: The 'a' coefficients of the filter
        :param num_channels: Number of channels of input data
        """
        self.b = b
        self.a = a
        self.num_channels = num_channels
        self.reset_zi(
            axis,
        )
        self.axis = axis

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        """

        :param batch: Input data to be filtered (n_channels x time_samples)
        :return filtered_batch: The filtered batch (same shape as batch)
        """

        # (Channels, Time)
        if len(batch.shape) == 1:
            batch = batch[:, None]
            squeeze = True
        else:
            squeeze = False
        filtered_batch, self.zi = lfilter(
            self.b, self.a, batch, axis=self.axis, zi=self.zi
        )
        return np.squeeze(filtered_batch) if squeeze else filtered_batch

    def reset_zi(self, axis: int = -1):
        """
        Reset the initial state of the filter
        """
        self.zi = np.swapaxes(
            np.asarray([lfiltic(self.b, self.a, np.zeros(self.num_channels))]), -1, axis
        )


class ButterworthFilter(LinearFilter):
    """
    Butterworth digital and analog filter class.

    """

    def __init__(
        self,
        num_channels: int,
        n: Union[int, float],
        wn: Union[list, float],
        btype: str = "low",
        analog: bool = False,
        fs: Optional[float] = None,
    ):
        """
        Initialize a linear Butterworth filter with the given parameters
        Parameters
        ----------
        :param num_channels: Number of channels
        :param n: Filter order
        :param wn: Cutoff frequencies (see scipy docs)
        :param btype: The type of filter
        :param analog: When True, return an analog filter, otherwise a digital
            filter is returned
        :param fs: The sampling frequency of the digital system
        """
        n = np.round(n)
        b, a = butter(n, wn, btype=btype, analog=analog, output="ba", fs=fs)
        super().__init__(b, a, num_channels)


class FIRFilter(LinearFilter):
    """
    FIR filter design using the window method.
    """

    def __init__(
        self,
        num_channels: int,
        numtaps: int,
        cutoff: list,
        width: Optional[float] = None,
        window: str = "hamming",
        pass_zero: bool = True,
        scale: bool = True,
        fs: Optional[float] = None,
    ):
        """

        :param num_channels: Number of channels
        :param numtaps: Length of the filter (number of coefficients)
        :param cutoff: Cutoff frequency of filter (expressed in the same units
            as `fs`)
        :param width: the approximate width
        of the transition region
        :param window: See `scipy.signal.get_window` for a list
        of windows and required parameters.
        :param pass_zero: {True, False, 'bandpass', 'lowpass', 'highpass',
            'bandstop'}
        :param scale: Set to True to scale the coefficients so that the
            frequency response is exactly unity at a certain frequency.
        :param fs: The sampling frequency of the signal.
        """
        b = firwin(
            numtaps,
            cutoff,
            width=width,
            window=window,
            pass_zero=pass_zero,
            scale=scale,
            fs=fs,
        )
        super().__init__(b, np.array([1.0]), num_channels)


class NotchFilter(LinearFilter):
    """
    Notch filter class.

    """

    def __init__(
        self, num_channels: int, w0: float, Q: Union[int, float], fs: float = 2, axis=-1
    ):
        """
        Initialize a linear Notch filter with the given parameters
        Parameters
        ----------
        :param num_channels: Number of channels
        :param w0: Frequency to remove from a signal
        :param Q: Quality factor. Dimensionless parameter that characterizes notch filter
        :param fs: The sampling frequency of the system
        """

        # b, a = iirnotch(w0, Q, fs=fs)
        b, a = butter(5, (w0 - 20, w0 + 20), btype="bandstop", fs=fs)
        super().__init__(b, a, num_channels, axis=axis)


class ButterworthBandPassFilterBank:
    """A cascade of butterworth filters:

    Given an input X with shape (channels, time) returns a series
    of band-passed version of X concatenated on the first axis
    (num_filters, channels, time) where num_filters is the number
    of used filters
    """

    def __init__(
        self,
        cut_offs: List[Tuple[float, float]],
        sample_frequency: float,
        order: int = 5,
    ):
        """
        Args:
            cut_offs: list of cut-off frequency in Hz defined by (low, high)
            sample_frequency: sample frequency in Hz
            order: order of the butterworth filter
        """
        self._filters = [
            butter(order, cf, "bandpass", fs=sample_frequency) for cf in cut_offs
        ]

    @property
    def n_filters(self) -> int:
        return len(self._filters)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: multivariate time series with shape (channels, timestamps)

        Returns:
            a tensor with shape (num_filters, channels, timestamps)
        """
        return np.stack([filtfilt(b, a, X, axis=1) for b, a in self._filters])
