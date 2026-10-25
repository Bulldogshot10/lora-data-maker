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

When you tap **scrape images** in the WebUI (or run a Playwright CLI scraper), the browser opens a persistent Chromium profile. On Termux, a visible window is used by default so you can sign in or finish a consent/verification challenge yourself. The browser profile keeps cookies and session state between runs; no Google password or cookie needs to be placed in the project config.

Default profiles and loopback CDP ports are:
- Google: $XDG_DATA_HOME/lora-data-maker/google-profile (normally ~/.local/share/lora-data-maker/google-profile), port 9222
- Bing: $XDG_DATA_HOME/lora-data-maker/bing-profile, port 9233
- WireGuard: separate persistent profiles named playwright-google-wireguard-profile or playwright-bing-wireguard-profile, so cookies are retained between proxied runs without reusing the non-proxy browser profile

For a custom Google profile, set CHROME_USER_DATA_DIR; for Bing use CHROME_BING_USER_DATA_DIR. The CDP ports can be changed with CHROME_DEBUG_PORT and CHROME_BING_DEBUG_PORT. CHROME_AUTO_LAUNCH=0 requires an already-running compatible browser. Set LORA_DATA_MAKER_HEADLESS=1 to force headless mode; manual challenge completion needs a visible window, so keep the default headed mode when using Google/WireGuard verification. On ordinary Linux without a display, headless mode is selected automatically unless DISPLAY is set.

The scraper uses the adapter's locator API to wait for dynamically rendered result cards and newly loaded cards after scrolling. Its network monitor logs significant browser-side HTTP errors and blocks common analytics/tracker endpoints only; it does not block page scripts, image downloads, or consent/verification resources. Google verification challenges pause the current job until you finish them in the visible Chromium window; the scraper then resumes the same search session. The WebUI's **cancel** action can stop a job that is waiting.

**Security:** CDP is a privileged browser-control endpoint. Keep the default CDP listeners on loopback (127.0.0.1) and never expose ports 9222/9233 on a LAN, through a tunnel, or via a reverse proxy. Treat the profile directories as sensitive because they contain login cookies and session state. The WebUI itself has no user login; leave it on localhost. See the repository's [Security Policy](../SECURITY.md).

~~~bash
# First use opens a visible persistent Chromium session on Termux:X11.
lora-data-maker scrape playwright-google "julia russian woman" --count 10
lora-data-maker scrape playwright-bing "julia russian woman" --count 10
~~~

To start the WebUI from the same X11-enabled Termux session, run:

~~~bash
./mk.sh
~~~

If Android prevents Termux:X11 from opening automatically, bring the Termux:X11 app to the foreground once and retry. To inspect the local Google CDP endpoint, run curl -s http://127.0.0.1:9222/json/version; Bing defaults to port 9233. Sign into the provider in its own profile once. A WireGuard-enabled run uses a separate profile, so sign into that proxied profile separately if the provider requires authentication.

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
