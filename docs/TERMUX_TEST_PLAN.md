# Termux validation plan

This project targets current Termux Python on aarch64 Android.

The GitHub CI matrix validates the portable Python test suite on Linux with Python 3.12 and 3.13. Device-specific validation should be run on the target Termux installation before treating native optional packages as supported on that device.

## Clean install

~~~bash
pkg update
pkg upgrade
pkg install python git
git clone https://github.com/Bulldogshot10/lora-data-maker.git
cd lora-data-maker
python -m pip install .
lora-data-maker --version
~~~

Current release line:

~~~text
1.3.x
~~~

## Core CLI smoke

~~~bash
lora-data-maker --help
lora-data-maker init ~/ldm-termux-smoke --name smoke --subject smoke
cd ~/ldm-termux-smoke
lora-data-maker info
lora-data-maker reconcile
lora-data-maker build --help
lora-data-maker export --help
lora-data-maker pack --help
~~~

## File pipeline smoke

Place a valid image in `raw/`, then run:

~~~bash
lora-data-maker reconcile
lora-data-maker validate
lora-data-maker clean
lora-data-maker process
~~~

For Chromium backend testing, install the Termux Chromium package and the `.[chromium]` extra. For WD14 tagging, install the `tagger` extra:

~~~bash
python -m pip install ".[tagger]"
lora-data-maker tag
lora-data-maker clean-tags
lora-data-maker caption
lora-data-maker export --repeats 1
lora-data-maker pack
~~~

## Incremental smoke

Add another raw image after the first successful run.

~~~bash
lora-data-maker reconcile
lora-data-maker validate
lora-data-maker clean
lora-data-maker process
~~~

Confirm that previously completed records remain at their later states.

Run another scrape using the same source query and confirm already-known source URLs are skipped.

## Native dependency gate

If ONNX Runtime cannot install or load correctly on the target Termux build, do not promote that device/runtime combination to a supported requirement. The core pipeline remains usable without the optional tagger runtime.

## Results

Record the device model, Android version, Termux version, Python version, optional runtime status, and any failures before considering a device-specific combination validated.