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

python -m pip install ".[quality]"
~~~

Check the CLI:

~~~bash
lora-data-maker --version
lora-data-maker --help
~~~

The core package uses Python standard library modules plus Pillow and requests.

## Optional Chromium backend

The existing Chromium scraper requires the Termux chromium package plus the Selenium extra:

~~~bash
pkg install chromium
python -m pip install ".[chromium]"
~~~

The backend expects `chromium-browser` and `chromedriver` on PATH.

## Optional Playwright image-search backend

Termux does not use the upstream Playwright browser bundle directly. The upstream Python package publishes Linux glibc wheels, while Termux uses Android/Bionic. The project therefore uses the Termux-specific `playwright-termux` adapter, which drives the native Termux Chromium through CDP.

Install it with:

~~~bash
pkg install x11-repo termux-x11-nightly chromium
python -m pip install ".[playwright]"
~~~

This installs the pinned `playwright-termux` 2.0.0 CDP adapter used by the backend.
Do not run `playwright install` for this backend.

## Optional Playwright WireGuard proxy

The WebUI can upload a WireGuard client `.conf` and use it only for Playwright scraping. It does not replace Android's global VPN or change the device's default route. Install the userspace proxy executable first:

~~~bash
pkg install golang
go install github.com/windtf/wireproxy/cmd/wireproxy@v1.1.2
export PATH="$HOME/go/bin:$PATH"
wireproxy --version
~~~

If the Go binary directory is not on PATH in future Termux sessions, add the PATH export to your shell profile. In WebUI → collect images, select a Playwright backend, upload the provider's WireGuard `.conf`, and enable **use WireGuard for Playwright scraping**. The private key is stored outside dataset folders with owner-only file permissions. The local WireGuard proxy is started for the scrape and stopped afterward. SafeSearch is a separate checkbox; it defaults off and can be enabled to filter explicit results.

Test Bing:

~~~bash
lora-data-maker scrape playwright-bing "julia russian woman" --count 10
~~~

Test Google:

~~~bash
lora-data-maker scrape playwright-google "julia russian woman" --count 10
~~~

When you tap **scrape images** in the WebUI (or run the Google CLI scraper), the Google backend opens the Termux:X11 Android app, starts its display, and launches a **visible Termux Chromium window** automatically if the debugging endpoint is not already running. Both the Termux:X11 Android app and the `termux-x11-nightly` package must be installed. It uses a persistent profile at `$XDG_DATA_HOME/lora-data-maker/google-profile` (normally `~/.local/share/lora-data-maker/google-profile`). Sign into Google in that window once; the profile keeps the local browser session for future searches. You do not need to put a Google password, cookie, or token in the project config.

If you already have a dedicated Chromium profile, set `CHROME_USER_DATA_DIR` to that profile before running the scraper. Set `CHROME_DEBUG_PORT` to change the local CDP port (default 9222), or `CHROME_AUTO_LAUNCH=0` to require an already-running browser. The backend only connects to a loopback CDP endpoint.

**Security:** CDP is a privileged browser-control endpoint. Use a dedicated Chromium profile for scraping, keep `--remote-debugging-address=127.0.0.1`, and never expose port 9222 on a LAN, through a tunnel, or via a reverse proxy. Treat the profile directory as sensitive because it contains login cookies and session state. The WebUI itself has no user login; leave it on localhost. See the repository's [Security Policy](../SECURITY.md).

~~~bash
# First use: this starts Termux:X11 and opens the visible Chromium window.
lora-data-maker scrape playwright-google "julia russian woman" --count 10
~~~

Sign into Google in the visible Chromium window. The scraper opens its search in another tab and reuses the same local profile on later runs. If Google blocks the first search while you are signing in, finish login and rerun the command. The profile and its cookies stay under your Termux home, outside the repository.

To start the WebUI from the same X11-enabled Termux session, run:

~~~bash
./mk.sh
~~~

Select `playwright-google` in the WebUI and tap **scrape images**. The WebUI starts Termux:X11 and opens the browser automatically when needed. The browser stays open after a scrape so the signed-in session can be reused. If Android prevents the app from being opened automatically, bring Termux:X11 to the foreground once and retry. Check `curl -s http://127.0.0.1:9222/json/version` when troubleshooting. Google verification/consent pages can still block scraping; this backend reuses your local signed-in session but does not automate or bypass verification challenges.

## Quality/text-filter runtime

The clean stage uses the PP-OCRv3 text detector when quality.text_check = true, which is the default for newly initialized projects. On regular Python environments install:

~~~bash
python -m pip install ".[quality]"
~~~

On Termux, use the native OpenCV package when the headless Python wheel is unavailable:

~~~bash
pkg install x11-repo opencv-python
~~~

The text detector model is cached under cache/text/ and checksum-verified.

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

## WebUI dataset explorer

Start the WebUI from any dataset project:

~~~bash
lora-data-maker web
~~~

The WebUI treats the active project's parent directory as the dataset workspace. It automatically lists sibling directories that contain both config.toml and manifest.json, so previously created dataset projects can be opened without restarting the server.

The explorer only discovers direct children of that workspace. Switching projects keeps an already-running pipeline job bound to the project where it started.

Deleting a dataset removes its entire project directory. The active project cannot be deleted, so open another dataset first.

## First project

~~~bash
lora-data-maker init ~/datasets/julia --name julia --subject julia
cd ~/datasets/julia
lora-data-maker build
~~~

For manual control, use the individual stage commands documented in CLI.md.
