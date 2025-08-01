import logging
import time
from threading import Thread

import numpy as np
from numpy.typing import NDArray
from pylsl import StreamInlet, local_clock, resolve_bypred

from lsl_xdf_now.generic.lsl_utils import get_lsl_predicate

MAX_TOTAL_SAMPLES_GUARD = 10000


class LSLIterator:
    """
    A class to iterate over an LSL stream, pulling data in chunks.
    """

    def __init__(
        self,
        stream_type: str = "EEG",
        stream_name: str = "",
        source_id: str = "",
        verbose: bool = False,
        out_key: str = "eeg",
        processing_flags: int = 0,
        timeout: float = 0.0,
        max_samples: int = 1024,
        epoch_duration: float = 10,
    ):
        """
        Args:
        stream_name (Optional) (str): if given, the resolve stream will look for matched stream-name and type. Defaults to "";
        stream_type, stream_name, source_id
            if given, the resolve stream will look for matched stream-name and type. Defaults to "";
        stream_type, stream_name, source_id
            The three LSL identifiers used to build the *resolve* predicate.
            Empty strings are treated as wildcards.
            See `.lsl_generic_utils.get_lsl_predicate`.
        timeout
            Per-chunk pull timeout (seconds).
            Keep **0.0** to make the inlet completely non-blocking.
        processing_flags
            Flags forwarded to `pylsl.StreamInlet`
            (e.g. ``pylsl.proc_clocksync``).
        max_samples
            Maximum number of samples to request in each
            `pylsl.StreamInlet.pull_chunk` call.
        epoch_duration
            Sliding window (in seconds) – samples older than
            ``local_clock() - epoch_duration`` are discarded immediately
            after each pull.
        """
        self.stream_type = stream_type
        self.stream_name = stream_name
        self.source_id = source_id
        self.predicate = get_lsl_predicate(
            stream_type=self.stream_type,
            stream_name=self.stream_name,
            source_id=self.source_id,
        )
        self.timeout = timeout
        self.max_samples = max_samples
        self.epoch_duration = epoch_duration
        self.processing_flags = processing_flags

        self._inlet_ready = False
        self.inlet = self._resolve_stream()
        # self._thread_search_stream()
        self.out_key = out_key
        self.nhits = 0
        self._last_sample = -1.0
        self._last_sent = time.time()
        self._count_channel = self._get_channel_in_lsl()
        self._verbose = verbose

    def _thread_search_stream(self):
        self.thread_search_lsl = Thread(
            target=lambda: self._resolve_stream(),
            daemon=True,
        )
        self.thread_search_lsl.start()

    def _resolve_stream(self):
        """
        Resolve an LSL stream matching the provided predicate (type, name, source_id).
        Logs all matching streams and uses the first match.

        Assign:
            self._inlet:    StreamInlet: Connected LSL inlet for the chosen stream.


        """

        logging.info(f"Resolving LSL stream with predicate: {self.predicate}")

        while True:
            try:
                candidate_streams = resolve_bypred(self.predicate, timeout=5)
                if not candidate_streams:
                    logging.warning("No matching LSL stream found, retrying...")
                    continue
                break
            except Exception as e:
                logging.warning(f"LSL resolve failed: {e}, retrying...")

        if len(candidate_streams) > 1:
            logging.warning(
                f"Multiple LSL streams matched the predicate. Using the first match."
            )
            for idx, stream in enumerate(candidate_streams, 1):
                logging.info(
                    f"{idx}) {stream.name()} — {stream.type()} {stream.source_id()}"
                )

        self._inlet = StreamInlet(
            candidate_streams[0], recover=True, processing_flags=self.processing_flags
        )
        logging.info(f"Connected to stream: {self.inlet_description}")
        # do not remove time_correction from self._resolve.
        # This is blocking only first time called (here).
        self._inlet.time_correction()
        self._flush_stream()
        self._inlet_ready = True
        logging.info(f"LSL inlet {self.inlet_name} is ready.")
        return self._inlet

    def _get_channel_in_lsl(self) -> int:
        """
        Get the number of channels in the LSL stream.
        If a count channel is specified, it returns that value.
        Otherwise, it returns the number of channels in the inlet info.

        Returns:
            int: Number of channels in the LSL stream.
        """
        if not self._inlet_ready:
            logging.info("Inlet not ready, resolving stream...")
            self._resolve_stream()
        if self._count_channel is not None:
            return self._count_channel
        return self.inlet.info().channel_count() if self.inlet else 0

    def pull(self) -> tuple[NDArray, NDArray]:
        """
        Acquire the newest chunk from the connected LSL inlet and write it
        into *data_dict*.

        The call is **non-blocking**; if no fresh data are available the node
        automatically disables its children and post-node for that iteration.

        Returns
        -------
        tuple[ndarray, ndarray]
            The same dictionary possibly extended with:

            * `ndarray = (channels, N)``
            * ndarray ― ``shape = (N,)``

            where *N* ≤ ``max_samples`` (may be 0 when no data arrived).

        Notes
        -----
        * *pull_chunk* and *time_correction* are called in a loop until the
          inlet is empty.  A hard guard of ``MAX_TOTAL_SAMPLES_GUARD`` samples
          prevents accidental infinite loops.
        * The **epoch** filter is applied *after* time-correction: samples with
          timestamps older than ``local_clock() - epoch_duration`` are skipped.
        * If no data are available, the method returns ``None, None``.
        """
        if not self._inlet_ready:
            logging.info("Inlet not ready")
            return None, None

        all_samples: list[list] = []
        all_timestamps: list = []

        current_time = local_clock()
        epoch_start = current_time - self.epoch_duration

        # Continuosly pull as long as data is available (non-blocking)
        while True:
            # time correction not blocking here (as already called first time in _resolve)
            time_correction = self._inlet.time_correction(timeout=0)
            samples, timestamps = self._inlet.pull_chunk(
                timeout=0.0, max_samples=self.max_samples
            )

            if not samples:
                break
            timestamps = [t + time_correction for t in timestamps]

            discarded_samples = 0
            for s, t in zip(samples, timestamps):
                if t < epoch_start:
                    discarded_samples += 1
                    continue
                all_samples.append(s)
                all_timestamps.append(t)

            if discarded_samples:
                logging.warning(
                    f"Discarded {discarded_samples} of a pull of {len(samples)=}"
                )
            if len(all_samples) > MAX_TOTAL_SAMPLES_GUARD:
                logging.warning(
                    f"Aborted read after {MAX_TOTAL_SAMPLES_GUARD} samples to avoid infinite loop."
                )
                break

        if not all_samples:
            logging.warning("No LSL data — skipping this iteration.")
            return None, None
        # Convert to numpy arrays
        all_samples = np.array(all_samples).T
        all_timestamps = np.array(all_timestamps)

        return all_samples, all_timestamps

    def __iter__(self):
        """
        Returns an iterator object.
        """
        while not self._inlet_ready:
            logging.info("Waiting for LSL inlet to be ready...")
            time.sleep(1)
        return self

    def __next__(self):
        """
        Pulls the next sample from the LSL inlet.
        Returns:
            dict: A dictionary with the key `self.out_key` and the pulled sample as value.
        """
        sample = self.pull()
        if sample is None:
            raise StopIteration
        return {self.out_key: sample}

    def _flush_stream(self) -> None:
        """
        Flushes buffered data from an LSL inlet by pulling until no more data appears.

        Args:
            inlet (StreamInlet): The LSL stream inlet to flush.
            max_iters (int): Number of consecutive empty polls before stopping.

        Returns:
            bool: True if the flush procedure completes.
        """

        logging.info(f"Flushing {self.inlet_name} ...")
        while True:
            sample, _ = self._inlet.pull_chunk()
            if not sample:
                break
        logging.info(f"Flushed {self.inlet_name}!")

    @property
    def inlet_name(self) -> str:
        return f"{self._inlet.info().name()}"

    @property
    def inlet_description(self) -> str:
        return f"{self._inlet.info().name()} @ {self._inlet.info().nominal_srate()} Hz"
