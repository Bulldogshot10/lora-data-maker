# Security Policy

## Reporting a vulnerability

Please report security issues privately through the repository's GitHub security reporting feature if it is enabled. Do not publish exploit details, session cookies, API keys, or personal dataset contents in a public issue. Include the affected version or commit, the impact, and a minimal reproduction that does not expose credentials.

## Security model and boundaries

lora-data-maker is a local dataset tool, not a multi-user service. The WebUI defaults to `127.0.0.1:8787` and does not provide user accounts or authentication. Its loopback peer, Host, and browser Origin checks reduce accidental remote access and cross-origin browser requests; they are not a substitute for an operating-system security boundary. Do not publish the WebUI, expose it through a reverse proxy or tunnel, or assume it is suitable for an untrusted shared machine. Keep it on loopback and run it under a user account that can access only the files you intend to process.

## API credentials

The WebUI can persist `GEMINI_API_KEY` and `OPENAI_API_KEY` in a per-user JSON file outside dataset directories. The default path is `~/.config/lora-data-maker/api-keys.json`, or `$XDG_CONFIG_HOME/lora-data-maker/api-keys.json` when `XDG_CONFIG_HOME` is set. `LORA_DATA_MAKER_API_KEY_FILE` overrides the path.

The file is plaintext, not encrypted. The application writes the credential file with owner-only permissions (`0600`) and creates its own configuration directory with owner-only permissions (`0700`). A caller-supplied existing parent directory is not forcibly chmodded. Keys are loaded into the WebUI process environment on startup and remain available to that process while the server runs. Use OS account isolation and disk protection appropriate to the device. Never commit the credential file, put keys in a dataset `config.toml`, paste keys into public issues, or share logs that may contain private data. The WebUI's **clear** action removes the selected saved key and unsets it from the running process.

## Authenticated Google scraping and Chromium CDP

The `playwright-google` backend can attach to an already-running Chromium through the Chrome DevTools Protocol (CDP) when `CHROME_USER_DATA_DIR` is set. CDP is a privileged browser-control interface: anyone who can reach the debugging port may be able to inspect pages, read browser state, and perform actions as the signed-in account.

Use a dedicated Chromium profile for scraping, and bind the debugging listener only to `127.0.0.1` (the documented default port is `9222`). Never bind CDP to `0.0.0.0`, a LAN interface, or a public address; do not expose it via port forwarding, a tunnel, or a reverse proxy. Treat the profile directory as sensitive because it contains session state and cookies. Do not upload it, commit it, or share it. Google may still require consent or verification, and the scraper does not bypass those challenges.

## Outbound requests and uploads

The application validates outbound HTTP(S) destinations, rejects private/reserved addresses for normal downloads, limits redirects, disables redirects on API caption requests, and avoids silently applying environment-configured proxies to credentialed API calls. URL/DNS checks are application-level checks and do not pin a resolved IP address to the subsequent socket connection. Use only trusted custom API endpoints, especially when configuring an OpenAI-compatible base URL.

Image upload requests are bounded; individual image uploads are limited to 25 MiB. VLM file uploads are limited to 2 GiB and check the GGUF magic bytes, which confirms the file format marker but does not prove a model is safe. Only load model files from sources you trust. Scraped images, archives, model files, and metadata are untrusted input and should be kept within a controlled workspace.

## Before publishing changes

Review diffs for credentials, cookies, local profiles, personal images, generated datasets, private endpoints, and debug output. Confirm `.env`, credential stores, browser profiles, model caches, raw/work/output directories, and archives are not tracked. Run the test suite for the exact commit you plan to publish; manually triggered or historical CI runs do not validate a later commit.
