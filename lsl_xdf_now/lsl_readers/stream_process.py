import time

import numpy as np

from lsl_xdf_now.data_processing.filters import PreprocessIIR
from lsl_xdf_now.lsl_readers.lsl_iterator import LSLIterator


class StreamProcess:
    def __init__(
        self,
        n_channels: int = 1,
        stream_type: str = "GSR",
        stream_name: str = "",
        stream_source_id: str = "",
        filter_params: dict = None,
    ):
        """_summary_

        Parameters
        ----------
        n_channels : int, optional
            desired channels, by default 1
        stream_type : str, optional
            lsl stream_type, by default "GSR"
        stream_name : str, optional
            lsl stream_name, by default ""
        stream_source_id : str, optional
            lsl stream_source_id, by default ""
        filter_params : dict, optional
            low_cut
            high_cut
            sfreq
            filter_order
            num_channels
            notch_filter
            notch_filter_width
            notch_filter_order, by default None

        Raises
        ------
        ValueError
            _description_
        """
        self.lsl_bufferizer = LSLIterator(
            stream_type=stream_type,
            stream_name=stream_name,
            source_id=stream_source_id,
        )
        self.n_channels = n_channels
        if self.n_channels > self.lsl_bufferizer._get_channel_in_lsl():
            raise ValueError(
                f"Requested {self.n_channels} channels, but only {self.lsl_bufferizer._get_channel_in_lsl()} are available in the LSL stream."
            )
        self.inlet = self.lsl_bufferizer.inlet
        self.filter_ = None
        if filter_params:
            self.filter_ = PreprocessIIR(num_channels=n_channels, **filter_params)

    def pull(self):
        """Pulls a chunk from the LSL stream, optionally filters it, and returns (data, timestamps)."""
        data, ts = self.lsl_bufferizer.pull()
        if data is None or ts is None or len(data) == 0:
            return None, None
        data = np.array(data)
        if self.filter_:
            data = self.filter_(data)
        return data, ts


if __name__ == "__main__":
    # Example usage
    stream_process = StreamProcess(
        n_channels=1,
        stream_type="GSR",
        stream_name="",
        filter_params={
            "low_cut": None,
            "high_cut": 8,
            "sfreq": 250,
            "filter_order": 2,
            "notch_filter": 50,
            "notch_filter_width": 5,
            "notch_filter_order": 2,
        },
    )

    while True:
        data, timestamps = stream_process.pull()
        if data is not None:
            print(f"Data: {data}, Timestamps: {timestamps}")
        else:
            print("No data available.")
            continue
        # Add a sleep or break condition to avoid infinite loop in real applications.
        time.sleep(0.1)
        if len(data) > 1000:  # Example condition to break the loop
            print("Breaking after 1000 samples.")
            break
