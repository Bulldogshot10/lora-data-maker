# Termux installation

## Requirements

Primary target: Termux on aarch64 Android with Python 3.12 or newer.
Regular Linux/Python environments are also supported.

Check the environment:

~~~bash
uname -m
python --version
~~~

## Install from the repository

~~~bash
pkg update
pkg upgrade
pkg install python git

git clone https://github.com/Bulldogshot10/lora-data-maker.git
cd lora-data-maker

python -m pip install .
~~~

Check the CLI:

~~~bash
lora-data-maker --version
lora-data-maker --help
~~~

The core package uses Python standard library modules plus Pillow and requests.

## Optional Chromium backend

The Chromium scraper requires the Termux chromium package plus the Selenium extra:

~~~bash
pkg install chromium
python -m pip install ".[chromium]"
~~~

The backend expects `chromium-browser` and `chromedriver` on PATH.

## Optional tagger runtime

The WD14 tagger uses the optional tagger dependency set:

~~~bash
python -m pip install ".[tagger]"
~~~

This installs NumPy and ONNX Runtime. They are loaded lazily, so the basic CLI can run without the tagger runtime.

On Termux, native optional packages depend on the target device and package availability. The core image and dataset stages do not require ONNX Runtime.

## Development install

~~~bash
python -m pip install ".[dev]"
pytest -q
~~~

The dev extra supplies NumPy and pytest.

## Android shared storage

Grant Termux storage access when working with Android shared storage:

~~~bash
termux-setup-storage
~~~

Shared storage is available below ~/storage/shared/.

Keeping active projects inside Termux home is usually simpler for repeated processing.

## First project

~~~bash
lora-data-maker init ~/datasets/julia --name julia --subject julia
cd ~/datasets/julia
lora-data-maker build
~~~

For manual control, use the individual stage commands documented in CLI.md.
