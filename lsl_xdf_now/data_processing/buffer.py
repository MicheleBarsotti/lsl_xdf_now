from typing import Optional, Union

import numpy as np


def parse_dtype(dtype: Union[str, np.dtype, type]) -> np.dtype:
    """
    Convert dtype input into a NumPy dtype object.

    Supports:
    - Strings (e.g., "float32", "int16")
    - NumPy dtype objects (e.g., np.dtype("float32"))
    - NumPy types (e.g., np.float32, np.int16)

    Raises:
        ValueError: If dtype is not a recognized format.
    """
    if isinstance(dtype, np.dtype):
        return dtype  # Already a valid dtype instance
    if isinstance(dtype, str):
        return np.dtype(dtype)  # Convert string to dtype
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return np.dtype(dtype)  # Convert np.float32 → np.dtype("float32")

    raise ValueError(
        f"Unsupported dtype format: {dtype} (expected str, np.dtype, or np.generic subclass)"
    )


class RingBuffer:
    """
    Implementation of a RingBuffer, a simplification of:
    https://github.com/eric-wieser/numpy_ringbuffer/blob/master/numpy_ringbuffer/__init__.py

    Create a new ring buffer with the given time_len and element type.
    The buffer will have shape: (C, T) where C are the number of features and T is the
    time length of the buffer;
    NOTE: We rather not inherit from ndarray to avoid the temptation to use methods without
        the proper rotation of the ring;
    """

    def __init__(
        self,
        time_len: int,
        channels: int = 1,
        dtype: Union[str, np.dtype] = "float32",  # Accepts both string and dtype
        stride_samples: Optional[int] = None,
    ):
        """
        Args:
        time_len: int
            Maximum number of time observations;
        channels: int
            Number of channels each time observations consists of;
        dtype: data-type, optional
            Desired type of buffer elements.
        stride_samples (Optional[int]): Number of new samples required to signal readiness.
                If None, defaults 1.
        """
        if channels <= 0:
            raise ValueError("`channels` must be at least 1;")
        if time_len <= 0:
            raise ValueError("`time_len` must be at least 1;")

        self._channels = channels
        self._time_len = time_len
        self._hits = 0
        self._new_samples = 0
        self._left_index = 0
        self._stride = stride_samples if stride_samples is not None else 1
        self._dtype = parse_dtype(dtype=dtype)

        self._arr = np.empty((channels, time_len), self._dtype)
        self._out_arr = np.empty_like(self._arr)

    def _unwrap(self) -> np.ndarray:
        """Copy the data from this buffer into unwrapped form"""
        arr_copied = self._arr.copy()
        if self._hits <= self._time_len:
            return arr_copied[..., : self._hits]
        if self._left_index == 0:
            self._out_arr = arr_copied
        else:
            self._out_arr[:, : -self._left_index] = arr_copied[:, self._left_index :]
            self._out_arr[:, -self._left_index :] = arr_copied[:, : self._left_index]
        return self._out_arr

    def _fix_indices(self):
        self._left_index = self._hits % self._time_len

    def __array__(self):
        """
        __array__ Returns the buffer with the proper rotation when np.asarray(buffer)
        """
        return self._unwrap()

    @property
    def values(self) -> np.ndarray:
        """
        Returns the buffer with the proper rotation;
        """
        return self._unwrap()

    @property
    def dtype(self):
        """
        Returns the dtype of the underlying array;
        """
        return self._arr.dtype

    @property
    def shape(self):
        """
        Returns the valid shape of the underlying array;
        """
        return *self._arr.shape[:-1], min(self._hits, self._time_len)

    @property
    def ready(self) -> bool:
        """
        Returns True if the buffer has at least time len elements;
        Initial transition
        """
        return self._hits >= self._time_len

    def consume_stride(self) -> Optional[np.ndarray]:
        """
        if ready and stride matched it returns the values and reset stride counter (new-values)
        otherwise it returns None
        """
        if self.stride_ready_and_reset():
            return self.values
        return None

    def stride_ready_and_reset(self) -> bool:
        """
        Returns True if the number of new samples added since the last window extraction
        is at least equal to the defined stride.

        NB:
            This method already consume_stride() (i.e., reset new-sample-counter)
                when stride is ready
        """
        if self.ready and self._new_samples >= self._stride:
            self.reset_stride()
            return True
        return False

    def reset_stride(self):
        """
        Call this method after processing a window to Reset the new_samples value
        Originally it subtracts the stride value from the new samples counter as follows:
            ```
            if self._new_samples >= self._stride:
                self._new_samples -= self._stride
            else:
                self._new_samples = 0
            ```
        """
        self._new_samples = 0

    def add(self, value: np.ndarray):
        """extend (if value.ndim>1) or append (value.ndim==1) data in the buffer"""
        if value.ndim > 1:
            self.extend(values=value)
        else:
            self.append(value=value)

    def append(self, value: np.ndarray):
        """
        Append elements to the buffer
        Args:
            value (np.ndarray): Must be of shape (C, ), where C is the number of channels;
        """
        if value.shape != (self._channels,):
            msg = "Can only `append` vector of shape (C, ),"
            msg += f" where C is the number of channels == {self._channels}"
            msg += f" instead input value.shape = {value.shape}"
            raise ValueError(msg)
        self._arr[:, self._left_index] = value
        self._hits += 1
        self._new_samples += 1
        self._fix_indices()

    def extend(self, values: np.ndarray):
        """Extend buffer: case in which input values have .ndim>1"""
        if values.shape[0] != self._channels:
            raise ValueError(
                f"Expected first dimension of values to be {self._channels}, got {values.shape[0]}"
            )

        n_new = values.shape[1]  # Number of new time observations

        if n_new >= self._time_len:
            # If more data is added than the buffer size, keep only the most recent part
            self._arr = values[:, -self._time_len :]
            self._hits = self._time_len  # useful in case of fix indices next time
            self._new_samples += n_new
            self._left_index = 0
        else:
            # Insert data, possibly wrapping around
            end_index = self._left_index + n_new

            if end_index <= self._time_len:
                # No wrapping needed
                self._arr[:, self._left_index : end_index] = values
            else:
                # Wrapping occurs, so split insertion into two slices
                first_part_size = self._time_len - self._left_index
                self._arr[:, self._left_index :] = values[:, :first_part_size]
                self._arr[:, : end_index - self._time_len] = values[:, first_part_size:]

            # Update hits and left index after batch append
            self._hits += n_new
            self._new_samples += n_new
            self._fix_indices()

    def extend_old_method(self, values: np.ndarray):
        """
        Legacy method (not efficient)
        Append sequence of elements to the buffer;
        Args:
            values (np.ndarray): Must be of shape (C, T) where C is the number of channels
            and T the number of time observations;
        """
        assert values.ndim == 2
        for time_observation in values.T:
            self.append(time_observation)

    def reset(self):
        """Reset method"""
        self._hits = 0
        self._new_samples = 0
        self._left_index = 0
        self._arr = np.empty(0)

    def __len__(self):
        return min(self._hits, self._time_len)

    def __iter__(self):
        return iter(self.values.T)

    def __repr__(self):
        return f"<RingBuffer of {np.asarray(self)}>"

    def percentage_ready(self) -> float:
        """return the percentage of buffer"""
        return np.round(self._hits / self._time_len * 100, 1)

    def percentage_stride_ready(self) -> float:
        return np.round(self._new_samples / self._stride * 100, 1)

    def message_readiness(self) -> str:
        if self.ready:
            return f"stride @ {self.percentage_stride_ready()} %"
        else:
            return f"buffer @ {self.percentage_ready()} %"
