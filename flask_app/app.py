import argparse
import math
import random
import threading
import time
from pathlib import Path

import yaml
from flask import Flask, jsonify, send_from_directory

from lsl_xdf_now.generic.lsl_utils import check_for_lsl_stream
from lsl_xdf_now.lsl_readers.stream_process import StreamProcess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--simulate",
    action="store_true",
    help="Simulate LSL input instead of using StreamProcess",
)
args = parser.parse_args()

if args.simulate:
    print(
        """
          SIMULATED SIGNAL!
          """
    )
    SIMULATED_SIGNAL = True

    class SimulatedStreamProcess:
        def __init__(self, **kwargs):
            self.start_time = time.time()

        def pull(self):
            t = time.time() - self.start_time
            # Sine wave with 5s period
            sine_value = 5 * math.sin(2 * math.pi * t / 10) + 5
            # Add small random noise
            noisy_value = sine_value + random.uniform(0, 0.5)
            return [[0, 0, noisy_value]], time.time()

    StreamProcessClass = SimulatedStreamProcess
else:
    StreamProcessClass = StreamProcess
    SIMULATED_SIGNAL = False


cfg_file = Path(__file__).absolute().parent / "stream_config.yaml"
app = Flask(__name__, static_folder="static")

feature_values = {}
stream_processes = {}

# Load stream configs at startup
with open(cfg_file) as f:
    stream_configs = yaml.safe_load(f)

# Start a StreamProcess for each config
for stream_id, cfg in enumerate(stream_configs):
    if not SIMULATED_SIGNAL and not check_for_lsl_stream(
        stream_type=cfg.get("stream_type", "GSR"),
        stream_name=cfg.get("stream_name", ""),
        source_id=cfg.get("stream_source_id", ""),
        timeout=cfg.get("timeout", 3),
    ):
        continue
    sp = StreamProcessClass(
        n_channels=1,
        stream_type=cfg.get("stream_type", "GSR"),
        stream_name=cfg.get("stream_name", ""),
        stream_source_id=cfg.get("stream_source_id", ""),
        filter_params=cfg.get("filter_params", {}),
    )
    stream_processes[stream_id] = sp
    feature_values[stream_id] = 0.0

    def update_feature(sid=stream_id, sproc=sp):
        while True:
            sample, ts = sproc.pull()
            if sample is not None and ts is not None:
                feature_values[sid] = sample[0][-1]
            time.sleep(0.1)

    threading.Thread(target=update_feature, daemon=True).start()


@app.route("/")
def index():
    return send_from_directory("static", "sound_ui.html")


@app.route("/feature/<int:stream_id>", methods=["GET"])
def get_feature(stream_id):
    return str(feature_values.get(stream_id, 0.0))


@app.route("/stream_configs", methods=["GET"])
def get_stream_configs():
    # Send aliases and stream_ids to frontend
    return jsonify(
        [
            {"idx": i, "alias": cfg.get("alias", f"Stream {i}")}
            for i, cfg in enumerate(stream_configs)
        ]
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
