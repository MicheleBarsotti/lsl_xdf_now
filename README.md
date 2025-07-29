# lsl_xdf_now

## Overview

**lsl_xdf_now** is a Python-based toolkit for real-time and offline processing of biosignal streams using [Lab Streaming Layer (LSL)](https://github.com/sccn/labstreaminglayer) and XDF files.  
It provides a modular backend for stream acquisition, feature extraction, and live feedback, as well as a modern web-based frontend for visualization and audio feedback.

This repository is designed for researchers and developers working with physiological signals (e.g., GSR, EEG) who need a flexible, extensible, and user-friendly environment for both online and offline experiments.

---

## Features

- **Real-time LSL stream acquisition** with configurable stream parameters.
- **Support for multiple streams** (e.g., GSR, EEG) with independent processing.
- **Feature extraction** and live value tracking for each stream.
- **Web-based UI** for visualization:
  - Oscilloscope and signal plots for each stream.
  - Per-stream controls for input/output mapping and audio feedback.
  - Audio feedback with frequency mapping based on signal features.
- **Configurable via YAML**: Easily define streams and their parameters in `stream_config.yaml`.
- **Pre-commit hooks** for code quality (Black, isort, mypy, pycln, etc.).

---

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/MicheleBarsotti/lsl_xdf_now.git
cd lsl_xdf_now
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
# For development:
pip install pre-commit
pre-commit install
```

### 3. Configure Streams

Edit `flask_app/stream_config.yaml` to define your streams.  
Example:

```yaml
- alias: "GSR1"
  stream_type: "GSR"
  stream_name: "Shimmer_GSR_FBF5_26"
  stream_source_id: ""
  filter_params:
    low_cut: null
    high_cut: 8
    sfreq: 250
    filter_order: 2
    notch_filter: 50
    notch_filter_width: 5
    notch_filter_order: 2
```

### 4. Run the Backend

```sh
cd flask_app
python app.py
```

The backend will start and serve the web UI at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

### 5. Open the Web UI

Open your browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

---

## Project Structure

```
lsl_xdf_now/
│
├── flask_app/
│   ├── app.py                # Flask backend
│   ├── static/
│   │   ├── sound_ui.html     # Web UI HTML
│   │   ├── sound_ui.js       # Web UI JS
│   │   ├── style.css         # Web UI CSS
│   │   └── logo.png
│   └── stream_config.yaml    # Stream configuration
│
├── lsl_xdf_now/
│   └── lsl_readers/
│       └── stream_process.py # StreamProcess class and helpers
│
├── requirements.txt
├── .pre-commit-config.yaml
└── README.md
```

---

## Development & Code Quality

- **Pre-commit hooks** are configured for formatting, linting, and static analysis.
- Run `pre-commit run --all-files` to check code quality manually.
- Contributions are welcome! Please open issues or pull requests.

---

## Acknowledgments

- Thanks to the [Hensis team](https://www.henesis.eu/) 
- Built on top of [Lab Streaming Layer (LSL)](https://github.com/sccn/labstreaminglayer) and [XDF](https://github.com/sccn/xdf).
- Thanks to the SCCN team and the open-source community.

---

## License

This project is licensed under the MIT License.
