"""
Useful functions for LSL stream handling
"""

import platform
from math import floor
from time import time
from typing import Sequence

from lsl_xdf_now.lsl_import import (
    IRREGULAR_RATE,
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    resolve_bypred,
)

source_uiid_suffix = f"{platform.node()}_{platform.system()}"


CHANNEL_FMT_CODE = {
    0: "undefined",
    1: "cf_float32",
    2: "cf_double64",
    3: "cf_string",
    4: "cf_int32",
    5: "cf_int16",
    6: "cf_int8",
    7: "cf_int64",
}


def stream_outlet_repr(stream_outlet: StreamOutlet):
    """attempt to create an repr of a streamOutlet, cannot access to its name and type (info)"""
    # ['obj', 'channel_format',
    #  'channel_count',
    #  'do_push_sample',
    #  'do_push_chunk',
    #  'value_type',
    #  'sample_type',
    #  '__module__',
    #  '__doc__',
    #  '__init__',
    #  '__del__',
    #  'push_sample',
    #  'push_chunk', 'have_consumers',
    #  'wait_for_consumers',
    #  '__dict__', '__weakref__', '__repr__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']
    msg = f"""
    have_consumers = {stream_outlet.have_consumers()}
    channel_count = {stream_outlet.channel_count}
    channel_format = {stream_outlet.channel_format}  ({CHANNEL_FMT_CODE[stream_outlet.channel_format]})
    """
    return msg


def resolve_stream_by_pred_construction(
    stream_name="", stream_type="", stream_sourceid="", timeout=5
):
    pred = get_lsl_predicate(
        stream_type=stream_type, stream_name=stream_name, source_id=stream_sourceid
    )
    streams = []
    while not streams:
        print(f"looking for stream '{pred}'")
        streams = resolve_bypred(predicate=pred, timeout=timeout)
    return streams


def get_lsl_predicate(stream_type: str, stream_name: str, source_id: str):
    pred = ""
    pred = f"{pred}type='{stream_type}' and " if stream_type else pred
    pred = f"{pred}name='{stream_name}' and " if stream_name else pred
    pred = f"{pred}source_id='{source_id}' and " if source_id else pred
    return " and ".join(pred.split(" and")[:-1]).strip()


def create_outlet(
    name: str,
    type_: str,
    channel_format: str,
    ch_label: Sequence[str] = [],
    ch_count: int = -1,
    nominal_srate: float = IRREGULAR_RATE,
    outlet_uuid: str = "",
) -> StreamOutlet:
    """
    Creates and returns an LSL StreamOutlet based on the given parameters.

    Args:
        name (str): The name of the stream.
        type_ (str): The type of data (e.g., "EEG", "EMG").
        channel_format (str): The format of the data (e.g., "float32", "int16").
        ch_label (Sequence[str], optional): Labels for each channel. Must be the same length as `ch_count` if provided. Defaults to an empty list.
        ch_count (int, optional): Number of channels. If not provided, it will be inferred from the length of `ch_label`. Defaults to -1.
        nominal_srate (float, optional): The nominal sampling rate. If the data is irregularly sampled, use IRREGULAR_RATE (default).
        outlet_uuid (str, optional): The unique source ID for the stream. If not provided, a default UUID is generated using the stream name, type, and a suffix.

    Returns:
        StreamOutlet: The created LSL StreamOutlet with the specified configuration.

    Raises:
        ValueError: If neither `ch_label` nor `ch_count` are provided or if they are inconsistent.

    Example:
        outlet = create_outlet(
            name="EEG_Stream",
            type_="EEG",
            channel_format="float32",
            ch_label=["Cz", "Pz", "Fz"],
            nominal_srate=500
        )
    """

    # Ensure a valid UUID is provided or generated
    if not outlet_uuid:
        outlet_uuid = f"{name}_{type_}_{source_uiid_suffix}"

    # Validate the channel count and labels
    if len(ch_label) == 0 and ch_count == -1:
        raise ValueError("At least one of 'ch_label' or 'ch_count' must be specified.")

    if ch_count != -1 and len(ch_label) > 0:
        if ch_count != len(ch_label):
            raise ValueError(
                f"Mismatch between ch_count ({ch_count}) and the number of channel labels ({len(ch_label)})."
            )

    elif ch_count == -1 and len(ch_label) > 0:
        # Infer the channel count from the number of labels
        ch_count = len(ch_label)

    elif ch_count != -1 and len(ch_label) == 0:
        # Generate default labels if none are provided
        ch_label = [f"ch{c}" for c in range(ch_count)]

    # Create the StreamInfo object for the LSL outlet
    stream_info = StreamInfo(
        name=name,
        type=type_,
        channel_count=ch_count,
        nominal_srate=nominal_srate,  # Default to irregular rate if not specified
        channel_format=channel_format,
        source_id=outlet_uuid,
    )

    # Append channel labels if available
    if ch_label:
        xml_channels = stream_info.desc().append_child("channels")
        for label in ch_label:
            xml_channels.append_child("channel").append_child_value("label", label)

    # Create and return the LSL StreamOutlet
    return StreamOutlet(stream_info)


def get_channel_name(inlet: StreamInlet):
    stream_info = inlet.info()
    stream_xml = stream_info.desc()
    chans_xml = stream_xml.child("channels")
    chan_xml_list = []
    ch = chans_xml.child("channel")
    while ch.name() == "channel":
        chan_xml_list.append(ch)
        ch = ch.next_sibling("channel")
    return [ch_xml.child_value("label") for ch_xml in chan_xml_list]


def check_for_lsl_stream(
    stream_type: str = "", stream_name: str = "", source_id: str = "", timeout=10
):
    """
    check_for_lsl_stream
    Create the predicate using  available arguments between (stream_type, stream_name, source_id) and
    check for it with "timeout"
    Args:
        stream_type (str, optional): _description_. Defaults to "".
        stream_name (str, optional): _description_. Defaults to "".
        source_id (str, optional): _description_. Defaults to "".
        timeout (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    time_start = time()
    cnt = 0
    max_tentatives = 4 if timeout >= 4 else floor(timeout)
    elapsed_time = time() - time_start
    streams = None
    lslpredicate = get_lsl_predicate(
        stream_name=stream_name, stream_type=stream_type, source_id=source_id
    )
    while elapsed_time < timeout:
        print(
            f"Does stream with pred '{lslpredicate}' exist?... (tentative {cnt} - timeout {timeout - elapsed_time:.1f})",
            end="\r",
        )
        cnt += 1
        elapsed_time = time() - time_start
        streams = resolve_bypred(
            predicate=lslpredicate, timeout=timeout // max_tentatives
        )
        if streams is not None and len(streams) > 0:
            print("\n----\n Stream Found!!\n")
            return True
    print(f"\n !!! -- \n Stream {stream_type} DOES NOT EXIST!!")
    return False
