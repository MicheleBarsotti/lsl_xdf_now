"""
Generic utils when working with xdf (multimodal) files
An xdf consists of "streams" that are
streams : list[dict] (one dict for each stream)
          Dicts have the following content:
          - 'time_series': Contains the time series as a [#Channels x #Samples]
            array of the type declared in ['info']['channel_format'].
          - 'time_stamps': Contains the time stamps for each sample (synced
            across streams).
          - 'info': Contains the meta-data of the stream (all values are
            strings).
          - 'name': Name of the stream.
          - 'type': Content type of the stream ('EEG', 'Events', ...).
          - 'channel_format': Value format ('int8', 'int16', 'int32', 'int64',
            'float32', 'double64', 'string').
          - 'nominal_srate': Nominal sampling rate of the stream (as declared
            by the device); zero for streams with irregular sampling rate.
          - 'effective_srate': Effective (measured) sampling rate of the
            stream if regular (otherwise omitted).
          - 'desc': Dict with any domain-specific meta-data.

In addition, the xdf contains fileheader that are:
    fileheader : Dict with file header contents in the 'info' field of streams.
"""

import contextlib
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from numpy.typing import NDArray
from pyxdf import load_xdf, resolve_streams

logger = logging.getLogger()


class _XDFInfoFooter:
    first_timestamp: Tuple[str]
    last_timestamp: Tuple[str]
    sample_count: Tuple[str]  # to be transformed in int
    clock_offsets: List[dict]


class _XDFFooter:
    info: _XDFInfoFooter


class XDFInfoStreamDict(TypedDict, total=True):
    name: List[str]
    type: List[str]
    channel_count: List[str]
    channel_format: List[
        Literal["int8", "int16", "int32", "int64", "float32", "double64", "string"]
    ]
    stream_id: int
    source_id: List[str]
    nominal_srate: List[str]
    effective_srate: Optional[List[str]]  # Optional list
    desc: List[Optional[Dict[str, str]]]
    version: List[str]
    created_at: List[str]
    uid: List[str]
    session_id: List[str]
    hostname: List[str]


class XDFStreamDict(TypedDict):
    time_series: NDArray  # Type depends on channel_format
    time_stamps: NDArray[np.float64]  # Always float64
    info: XDFInfoStreamDict
    footer: Optional[_XDFFooter]


def get_xdf_stream_infos(xdf_file_name: str) -> List[XDFInfoStreamDict]:
    """
    kept for legacy reason -> The good stuff is that it returns typed dict ;)
    Call the pyxdf.resolve_streams(xdf_file_name)

    Args:
        xdf_file_name (str): Name of the XDF file

    Returns:
        List[Dict[str, str]]: List of dicts containing information on each stream.
    """
    return resolve_streams(xdf_file_name)


def get_infostream_description(info: XDFInfoStreamDict) -> str:
    """Return a string with the information inside a stream["info"] dict

    Args:
        info (Dict[str, str]): stream-info

    Returns:
        str: verbose description of the stream info
    """
    return (
        f"{info['stream_id']}) {info['name']}, {info['type']}, {info['source_id']}  -  "
        + f"{info['channel_count']} {info['channel_format']} @{info['nominal_srate']}Hz"
    )


def filter_stream_info(
    infos: List[XDFInfoStreamDict],
    source_id: str = "",
    name: str = "",
    type_: str = "",
) -> List[XDFInfoStreamDict]:
    """filter_info_per_source_id:
    infos is the output of pyxdf.resolve_info(xdf_file)
    A single dict in infos has these keys:
    info keys:
    (['stream_id', 'name', 'type_', 'source_id', 'created_at',
        'uid', 'session_id', 'hostname', 'channel_count',
        'channel_format', 'nominal_srate'])
    Args:
        infos (List[dict]): list of stream infos dict
        source_id (str): string indicating the source id
        name (str): string indicating the name
        type (str): string indicating the type

    note:
     The "not" in front of "source_id", "name", and "type_" in the conditions means
     that if these arguments are empty strings (which can indicate they're not specified
     or not needed for filtering) if source_id is an empty string
     (indicating no filtering by source ID is desired),
     the condition not source_id will be True, and the corresponding check
     for info.get("source_id") == source_id will be skipped, effectively not filtering by source_id.
    """
    out_infos = []
    for info in infos:
        if (
            (not source_id or info.get("source_id") == source_id)
            and (not name or info.get("name") == name)
            and (not type_ or info.get("type") == type_)
        ):
            out_infos.append(info)
    return out_infos


def get_stream_channel_count(stream: XDFStreamDict) -> int:
    return int(stream["info"]["channel_count"][0])


def get_stream_channel_format(stream: XDFStreamDict) -> str:
    return stream["info"]["channel_format"][0]


def get_stream_name(stream: XDFStreamDict) -> str:
    return stream["info"]["name"][0]


def get_stream_type(stream: XDFStreamDict) -> str:
    return stream["info"]["type"][0]


def _get_stream_channel_x(
    stream: XDFStreamDict,
    field: Literal["channel", "unit", "impedance", "type", "label"],
    on_error_return: Literal["none", "indexes"] = "indexes",
    verbose: bool = True,
):
    try:
        return [
            dd[field][0] for dd in stream["info"]["desc"][0]["channels"][0]["channel"]
        ]
    except Exception as e:
        if verbose:
            print(f"no {field} found because of {e}.")
        if on_error_return == "indexes":
            if verbose:
                print("Returning numbers as labels ")
            return [f"ch{chix}" for chix in range(get_stream_channel_count(stream))]
        if verbose:
            print("Returning None")
        return None


def get_stream_labels(
    stream,
    on_error_return: Literal["none", "indexes"] = "indexes",
    verbose: bool = True,
):
    return _get_stream_channel_x(
        stream=stream, field="label", on_error_return=on_error_return, verbose=verbose
    )


def get_stream_channel_type(
    stream, on_error_return: Literal["none", "indexes"] = "none", verbose: bool = True
):
    return _get_stream_channel_x(
        stream=stream, field="type", on_error_return=on_error_return, verbose=verbose
    )


def get_stream_channel_unit(
    stream, on_error_return: Literal["none", "indexes"] = "none", verbose: bool = True
):
    return _get_stream_channel_x(
        stream=stream, field="unit", on_error_return=on_error_return, verbose=verbose
    )


def get_stream_channel_impedance(
    stream, on_error_return: Literal["none", "indexes"] = "none", verbose: bool = True
):
    return _get_stream_channel_x(
        stream=stream,
        field="impedance",
        on_error_return=on_error_return,
        verbose=verbose,
    )


def get_stream_nominal_srate(stream):
    return float(stream["info"]["nominal_srate"][0])


def get_stream_effective_srate(stream):
    return float(stream["info"]["effective_srate"])


def get_stream_uid(stream):
    return stream["info"]["uid"][0]


def get_stream_source_id(stream):
    return stream["info"]["source_id"][0]


def get_dict_stream_description(stream) -> Dict[str, Union[str, int, float]]:
    """get_dict_stream_description

    Args:
        stream (_type_): stream

    Returns:
        Dict[str, Union[str, int, float]]: descriptive dictionary of the stream
    """
    name = get_stream_name(stream)
    type_ = get_stream_type(stream)
    channel_frmt = get_stream_channel_format(stream)
    channel_n = get_stream_channel_count(stream)
    len_timestamps = len(stream["time_stamps"])
    labels = get_stream_labels(stream)
    nominal_srate = get_stream_nominal_srate(stream)
    effective_srate = get_stream_effective_srate(stream)
    uid = get_stream_uid(stream)
    source_id = get_stream_source_id(stream)

    out = {
        "stream_name": name,
        "stream_type": type_,
        "format": channel_frmt,
        "shape": f"({channel_n}, {len_timestamps})",
        "labels": labels,
        "n_channels": channel_n,
        "n_samples": len_timestamps,
        "nominal_rate": f"{nominal_srate:.2f}",
        "effective_rate": f"{effective_srate:.2f}",
        "uid": uid,
        "source_id": source_id,
    }
    if channel_frmt == "string":
        out["uniques_vals"] = np.unique(stream["time_series"]).tolist()
    if stream["time_stamps"].any():
        out["Duration"] = stream["time_stamps"][-1] - stream["time_stamps"][0]
        out["Start"] = stream["time_stamps"][0]
        out["End"] = stream["time_stamps"][-1]
    else:
        set_keys_as_none(out, keys=["Duration", "Start", "End"])
    _add_info_stream_statistic(stream, out)

    return out


def _add_info_stream_statistic(stream, out):
    try:
        prc_50_25_75 = np.percentile(stream["time_series"], [50, 25, 75])
        out["min_val"] = min(stream["time_series"])
        out["max_val"] = max(stream["time_series"])
        out["median"] = prc_50_25_75[0]
        out["prct25"] = prc_50_25_75[1]
        out["prct75"] = prc_50_25_75[2]
    except Exception as e:
        set_keys_as_none(out, keys=["min_val", "max_val", "median", "prct25", "prct75"])


def set_keys_as_none(out, keys: list):
    for key in keys:
        out[key] = None


def get_stream_description(
    stream: XDFStreamDict, ix: Optional[int] = None, short=False
) -> str:
    """
    verbose description of a stream
    """
    channel_n = get_stream_channel_count(stream)
    channel_frmt = get_stream_channel_format(stream)
    name = get_stream_name(stream)
    type_ = get_stream_type(stream)
    labels = get_stream_labels(stream)
    chan_types = get_stream_channel_type(stream)
    chan_unit = get_stream_channel_unit(stream)
    len_timestamps = len(stream["time_stamps"])
    nominal_srate = get_stream_nominal_srate(stream)
    effective_srate = get_stream_effective_srate(stream)
    uid = get_stream_uid(stream)
    source_id = get_stream_source_id(stream)

    out_txt = "Stream"
    ix_str = str(ix) or " "
    out_txt = f" {ix_str}:" if ix_str else ":"
    out_txt += f"{name} - type {type_} -"
    if short:
        return out_txt
    out_txt += f"""
        format {channel_frmt} -
        shape ({channel_n}, {len_timestamps})   at {nominal_srate:.1f} Hz  (effective {effective_srate:.1f} Hz)
        labels: {labels}
        chan_types: {chan_types}
        chan_unit: {chan_unit}
        uid: {uid}
        source_id", {source_id}"""
    if channel_frmt == "string":
        out_txt += f"string uniques-vals: {np.unique(stream['time_series']).tolist()}"
    if stream["time_stamps"].any():
        out_txt += f'\n\tDuration: {stream["time_stamps"][-1] - stream["time_stamps"][0]:.2f} s'
        out_txt += f'\n\tStart at: {stream["time_stamps"][0]:.2f} s'
        out_txt += f'\n\tEnd at: {stream["time_stamps"][-1]:.2f} s'
    else:
        out_txt = "\n\tno time stamps"
    if len(stream["time_series"]) > 0 and len(stream["time_series"][0]) == 1:
        with contextlib.suppress(Exception):
            out_txt += f'\n\tmin-max [{min(stream["time_series"])}-{max(stream["time_series"])}]'
            prc_50_25_75 = np.percentile(stream["time_series"], [50, 25, 75])
            out_txt += (
                f"\n\tmedian [{prc_50_25_75[0]} [{prc_50_25_75[1]} - {prc_50_25_75[2]}]"
            )
    else:
        out_txt += "\n\tno timeseries"
    return out_txt


def get_streams_dataframe(streams: List[XDFStreamDict]) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame with stream information.
    """
    out = []
    for stream in streams:
        out.append(get_dict_stream_description(stream))

    return pd.DataFrame(out)  # Create DataFrame from the dictionary


def print_stream_info(stream: XDFStreamDict, ix: Optional[int] = None, short=False):
    print(get_stream_description(stream=stream, ix=ix, short=short))


def get_stream_with_name(streams: List[dict], name: str) -> dict:
    """Return the stream that stream["info"]["name"][0] == name

    Args:
        streams (List[dict]): list of streams
        name (str): desired stream name

    Raises:
        RuntimeError: if more than 1 stream has the same name

    Returns:
        dict: stream dict
    """
    stream = [stream for stream in streams if stream["info"]["name"][0] == name]
    return __return_stream(stream=stream, stream_name=name)


def get_stream_with_type(streams: List[dict], stream_type: str) -> dict:
    """Return the stream that stream["info"]["type"][0] == type

    Args:
        streams (List[dict]): list of streams
        stream_type (str): desired stream type

    Raises:
        RuntimeError: if more than 1 stream has the same name

    Returns:
        dict: stream dict
    """
    stream = [stream for stream in streams if stream["info"]["type"][0] == stream_type]

    return __return_stream(stream=stream, stream_type=stream_type)


def __return_stream(
    stream: list,
    stream_type: str = None,
    stream_name=None,
):
    if len(stream) > 1 or not stream:
        logger.warning(
            f"Found {len(stream)} streams with {stream_type=}  {stream_name=}"
        )
        return stream
    return stream[0]


def get_index_channel_with_label(stream: dict, label: str) -> int:
    """get_index_channel_with_label
    Returns the index (int) of the channel with label == label


    Args:
        stream (dict): xdf-stream dict
        label (str): name of the channel

    Returns:
        int: index of the channel with the given label
    """
    labels = get_stream_labels(stream)
    return [idx for idx, desc in enumerate(labels) if desc == label][0]


def get_stream_channel_with_label(stream: dict, label: str) -> NDArray:
    """get_stream_channel_with_label
    Returns the channel (NDArray) with label == label

    Args:
        stream (dict): xdf-stream dict
        label (str): name of the channel

    Returns:
        NDArray: channel with the given name
    """
    ch_idx = get_index_channel_with_label(stream, label)
    return np.squeeze(stream["time_series"][:, ch_idx])


def write_timestamped_markers(
    ax: plt.Axes,
    time_series: Union[list, np.ndarray],
    time_stamps: np.ndarray,
    y_height: float = 0.5,
):
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    time_series = np.array(time_series).ravel()
    txts = [
        ax.text(s=ss, x=tt, y=y_height, rotation=60, transform=trans)
        for ss, tt in zip(time_series, time_stamps)
    ]


def plot_stream_list(streams, max_plot_fig: int = 8, max_number_legend=6):
    len_streams = len(streams)
    print(f"total streams: {len_streams}")
    if len_streams > max_plot_fig:
        print(f"Exeeded: {max_plot_fig} streams")
        for cnt in range(0, len_streams, max_plot_fig):
            eight_stream = (
                streams[cnt : cnt + max_plot_fig]
                if (cnt + max_plot_fig) < len_streams
                else streams[cnt:]
            )
            plot_stream_list(eight_stream)
        return
    fig, axx = plt.subplots(len(streams), sharex=True, figsize=(12, 12))
    if len(streams) == 1:
        axx = [axx]
    min_max_ts = [np.inf, -np.inf]
    for stream, ax in zip(streams, axx):
        min_max_ts = plot_single_stream(stream, ax, min_max_ts, max_number_legend)

        ax.legend(loc="upper right")  # type: ignore
    if not any(np.isinf(min_max_ts)):
        ax.set_xlim(min_max_ts[0], min_max_ts[1])  # type: ignore

    return fig, axx


def plot_single_stream(
    stream,
    ax: Optional[plt.Axes] = None,
    min_max_ts=[np.inf, -np.inf],
    max_number_legend=6,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))
    ch_cnt = get_stream_channel_count(stream=stream)
    ch_frmt = get_stream_channel_format(stream=stream)
    title = f"{get_stream_name(stream)}-{get_stream_type(stream)} ({ch_cnt}{ch_frmt})"
    labels = get_stream_labels(stream=stream)
    ax.set_title(title)  # type: ignore
    time_stamps = stream["time_stamps"]
    if not len(time_stamps):
        print(f"skip: {title} \n because no timestamp")
        return min_max_ts
    min_ts = min(min_max_ts[0], min(time_stamps))
    max_ts = max(min_max_ts[1], max(time_stamps))
    time_series = np.array(stream["time_series"])

    if ch_frmt == "string":
        if ch_cnt > 1:
            print(f"Warn: only first channel of {title} is shown")
            time_series = [tt[0] for tt in time_series]
        write_timestamped_markers(ax, time_series, time_stamps, y_height=0.2)  # type: ignore

    else:
        lines = ax.plot(time_stamps, time_series)  # type: ignore
        cnt = 0
        tot_len = len(lines)
        for line, label in zip(lines, labels):
            if cnt >= max_number_legend:
                line.set_label(f"... and other {tot_len - cnt}")
                break
            line.set_label(label)
            cnt += 1
    return [min_ts, max_ts]


def load_xdfstream_from_streaminfo(
    xdf_filepath: Union[str, Path],
    info: XDFInfoStreamDict,
    verbose: bool = False,
    dejitter_timestamps: bool = False,
    synchronize_clocks: bool = True,
) -> dict:
    """
    load_xdfstream_from_streaminfo

    Load one stream by using its stram_id extracted from the "info" dict passed.

    The desired info dict can be selected by using filter_stream_info passing the
     infos as extracted from pyxdf.resolve_info(xdf_file)

    Args:
        xdf_filepath (Union[str, Path]): xdf filepath
        info (dict): one of the instances of infos extracted
             from pyxdf.resolve_info(xdf_filepath)
        verbose (bool): load_xdf keyargs. Defaults to False.
        dejitter_timestamps (bool): load_xdf keyargs. Defaults to False.
        synchronize_clocks (bool): load_xdf keyargs. Defaults to True.

    Returns:
        dict: _description_
    """
    stream_id = info["stream_id"]
    stream = load_xdf(
        xdf_filepath,
        select_streams=stream_id,
        verbose=verbose,
        dejitter_timestamps=dejitter_timestamps,
        synchronize_clocks=synchronize_clocks,
    )
    return stream[0][0]
