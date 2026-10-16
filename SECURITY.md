# Security Policy

## Reporting a vulnerability

Please report security issues privately through the repository's GitHub security reporting feature if it is enabled. Do not publish exploit details, session cookies, API keys, or personal dataset contents in a public issue. Include the affected version or commit, the impact, and a minimal reproduction that does not expose credentials.

## Security model and boundaries

lora-data-maker is a local dataset tool, not a multi-user service. The WebUI defaults to `127.0.0.1:8787` and does not provide user accounts or authentication. Its loopback peer, Host, and browser Origin checks reduce accidental remote access and cross-origin browser requests; they are not a substitute for an operating-system security boundary. Do not publish the WebUI, expose it through a reverse proxy or tunnel, or assume it is suitable for an untrusted shared machine. Keep it on loopback and run it under a user account that can access only the files you intend to process.

## API credentials

The WebUI can persist `GEMINI_API_KEY` and `OPENAI_API_KEY` in a per-user encrypted vault outside dataset directories. The default path is `~/.config/lora-data-maker/api-keys.vault`, or `$XDG_CONFIG_HOME/lora-data-maker/api-keys.vault` when `XDG_CONFIG_HOME` is set. `LORA_DATA_MAKER_API_KEY_FILE` overrides the vault path.

The vault uses AES-256-GCM authenticated encryption, with a key derived from a user passphrase via scrypt. New passphrases must be at least 12 characters. The passphrase is never persisted; the vault must be unlocked each time the WebUI process starts. The encrypted file is written with owner-only permissions (`0600`), and the app-created config directory uses `0700` (an existing caller-supplied parent is not forcibly chmodded). On first successful unlock, an old `api-keys.json` plaintext store is migrated into the encrypted vault and then unlinked; unlinking cannot guarantee physical-media erasure, so disk-level encryption remains useful. While the vault is unlocked, decrypted keys are present in the WebUI process environment and memory. Lock the vault when finished, and use OS account isolation. The vault protects credentials at rest, not against malware or another process running as the same OS user while the vault is unlocked. Never commit credential files, put keys in a dataset `config.toml`, paste keys into public issues, or share logs that may contain private data. The WebUI's **clear** action removes the selected saved key from the encrypted vault and restores the pre-existing shell environment value, if any.

## Authenticated Google scraping and Chromium CDP

The `playwright-google` backend can attach to an already-running Chromium through the Chrome DevTools Protocol (CDP) when `CHROME_USER_DATA_DIR` is set. CDP is a privileged browser-control interface: anyone who can reach the debugging port may be able to inspect pages, read browser state, and perform actions as the signed-in account.

Use a dedicated Chromium profile for scraping, and bind the debugging listener only to `127.0.0.1` (the documented default port is `9222`). Never bind CDP to `0.0.0.0`, a LAN interface, or a public address; do not expose it via port forwarding, a tunnel, or a reverse proxy. Treat the profile directory as sensitive because it contains session state and cookies. Do not upload it, commit it, or share it. Google may still require consent or verification, and the scraper does not bypass those challenges.

## Outbound requests and uploads

The application validates outbound HTTP(S) destinations, rejects private/reserved addresses for normal downloads, limits redirects, disables redirects on API caption requests, and avoids silently applying environment-configured proxies to credentialed API calls. URL/DNS checks are application-level checks and do not pin a resolved IP address to the subsequent socket connection. Use only trusted custom API endpoints, especially when configuring an OpenAI-compatible base URL.

Image upload requests are bounded; individual image uploads are limited to 25 MiB. VLM file uploads are limited to 2 GiB and check the GGUF magic bytes, which confirms the file format marker but does not prove a model is safe. Only load model files from sources you trust. Scraped images, archives, model files, and metadata are untrusted input and should be kept within a controlled workspace.

## Before publishing changes

Review diffs for credentials, cookies, local profiles, personal images, generated datasets, private endpoints, and debug output. Confirm `.env`, credential stores, browser profiles, model caches, raw/work/output directories, and archives are not tracked. Before changing repository visibility, clone/fetch the full history and run `gitleaks git --redact --log-opts="--all" .` to scan all locally available refs; inspect every finding, rotate any exposed credential, and remove secret-bearing history where necessary. Deleting a secret from the latest file does not remove it from Git history. This audit did not run a full-history secret scan because a complete local clone/history was not available here. Run the test suite for the exact commit you plan to publish; manually triggered or historical CI runs do not validate a later commit.
