from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any


API_KEY_ENV_NAMES = ("GEMINI_API_KEY", "OPENAI_API_KEY")
_VAULT_FORMAT = "lora-data-maker-credential-vault"
_VAULT_VERSION = 1
_VAULT_AAD = b"lora-data-maker:credential-vault:v1"
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 128 * 1024 * 1024
_MIN_PASSPHRASE_LENGTH = 12
_MAX_VAULT_BYTES = 1024 * 1024


class CredentialVaultError(ValueError):
    """Raised for invalid, locked, or undecryptable credential vaults."""


def _credential_file_path(value: Path | str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise CredentialVaultError("credential vault paths must not be symlinks")
    if not raw.name or raw.name in {".", ".."}:
        raise CredentialVaultError("credential vault path must name a file")
    return raw.parent.resolve() / raw.name


def default_api_key_store_path() -> Path:
    """Return the per-user encrypted credential vault path."""
    override = os.environ.get("LORA_DATA_MAKER_API_KEY_FILE", "").strip()
    if override:
        return _credential_file_path(override)

    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return _credential_file_path(base / "lora-data-maker" / "api-keys.vault")


def default_legacy_api_key_store_path(vault_path: Path | str | None = None) -> Path | None:
    """Find the prior plaintext store location for one-time migration."""
    if os.environ.get("LORA_DATA_MAKER_API_KEY_FILE", "").strip():
        # Custom paths may themselves contain the old plaintext JSON. The vault
        # migrates it in place after a successful authenticated encryption.
        return None
    target = _credential_file_path(vault_path) if vault_path is not None else default_api_key_store_path()
    return target.with_name("api-keys.json")


def _crypto():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise CredentialVaultError(
            'encrypted API-key storage requires the "cryptography" package; '
            'install the WebUI extra with: python -m pip install ".[web]"'
        ) from exc
    return AESGCM


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    raw = passphrase.encode("utf-8")
    if not raw or len(raw) > 4096:
        raise CredentialVaultError("vault passphrase must contain between 1 and 4096 UTF-8 bytes")
    try:
        return hashlib.scrypt(
            raw,
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            maxmem=_SCRYPT_MAXMEM,
            dklen=32,
        )
    except (ValueError, MemoryError) as exc:
        raise CredentialVaultError("could not derive the vault encryption key") from exc


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise CredentialVaultError(f"vault field {field!r} is invalid")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as exc:
        raise CredentialVaultError(f"vault field {field!r} is invalid") from exc


def _read_legacy_json(path: Path) -> dict[str, str]:
    try:
        if path.stat().st_size > _MAX_VAULT_BYTES:
            raise CredentialVaultError("existing credential file exceeds the migration size limit")
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CredentialVaultError(
            f"could not read the existing credential file for migration: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise CredentialVaultError("existing credential file has an invalid format")
    return {
        name: value.strip()
        for name in API_KEY_ENV_NAMES
        if isinstance((value := raw.get(name)), str) and value.strip()
    }


def _atomic_write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".api-keys-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


class CredentialVault:
    """Passphrase-encrypted credential vault. The passphrase is never persisted.

    A derived AES-256 key is kept only while unlocked. Vault contents are
    encrypted with AES-GCM and a fresh nonce per save; scrypt derives the key
    from the user's passphrase. The first successful unlock migrates legacy
    plaintext JSON stores into the encrypted vault.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        legacy_path: Path | str | None = None,
    ) -> None:
        self.path = _credential_file_path(path)
        self.legacy_path = (
            _credential_file_path(legacy_path)
            if legacy_path is not None
            else None
        )
        self._key: bytes | None = None
        self._salt: bytes | None = None
        self._keys: dict[str, str] = {}

    @property
    def is_unlocked(self) -> bool:
        return self._key is not None

    @property
    def has_vault(self) -> bool:
        return self.path.is_file() and self._looks_encrypted(self.path)

    @property
    def has_legacy_file(self) -> bool:
        candidates = [self.path]
        if self.legacy_path is not None and self.legacy_path != self.path:
            candidates.append(self.legacy_path)
        return any(path.is_file() and not self._looks_encrypted(path) for path in candidates)

    def status(self) -> dict[str, bool]:
        return {
            "configured": self.has_vault,
            "unlocked": self.is_unlocked,
            "legacy_found": self.has_legacy_file,
        }

    @staticmethod
    def _looks_encrypted(path: Path) -> bool:
        try:
            # Files declaring the size limit or failing JSON parsing are handled
            # as non-migratable, rather than read into memory without a bound.
            if path.stat().st_size > _MAX_VAULT_BYTES:
                return True
            with path.open("r", encoding="utf-8") as handle:
                envelope = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        # Any file declaring a vault format must be handled as a vault, even
        # when its version is unsupported. Never reinterpret a damaged/newer
        # encrypted vault as legacy plaintext and overwrite it.
        return isinstance(envelope, dict) and "format" in envelope

    @staticmethod
    def _validate_keys(raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            raise CredentialVaultError("credential vault payload has an invalid format")
        result: dict[str, str] = {}
        for name in API_KEY_ENV_NAMES:
            value = raw.get(name)
            if isinstance(value, str) and value.strip():
                result[name] = value.strip()
        return result

    def _read_envelope(self, passphrase: str) -> tuple[bytes, bytes, dict[str, str]]:
        AESGCM = _crypto()
        try:
            if self.path.stat().st_size > _MAX_VAULT_BYTES:
                raise CredentialVaultError("encrypted credential vault exceeds the size limit")
            envelope = json.loads(self.path.read_text(encoding="utf-8"))
        except CredentialVaultError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CredentialVaultError("could not read encrypted credential vault") from exc
        if not isinstance(envelope, dict) or envelope.get("format") != _VAULT_FORMAT or envelope.get("version") != _VAULT_VERSION:
            raise CredentialVaultError("credential file is not a supported encrypted vault")
        expected_kdf = {"name": "scrypt", "n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P}
        if envelope.get("kdf") != expected_kdf:
            raise CredentialVaultError("credential vault uses an unsupported key-derivation configuration")
        salt = _b64decode(envelope.get("salt"), "salt")
        nonce = _b64decode(envelope.get("nonce"), "nonce")
        ciphertext = _b64decode(envelope.get("ciphertext"), "ciphertext")
        if len(salt) != 16 or len(nonce) != 12 or len(ciphertext) > 1024 * 1024:
            raise CredentialVaultError("credential vault data has an invalid size")
        key = _derive_key(passphrase, salt)
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, _VAULT_AAD)
        except Exception as exc:
            # Do not leak details that could distinguish a bad password from a
            # modified/tampered ciphertext.
            raise CredentialVaultError("incorrect passphrase or encrypted vault is damaged") from exc
        try:
            data = json.loads(plaintext.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CredentialVaultError("decrypted credential vault payload is invalid") from exc
        return key, salt, self._validate_keys(data)

    def _write_envelope(self, keys: dict[str, str]) -> None:
        if self._key is None or self._salt is None:
            raise CredentialVaultError("credential vault is locked")
        AESGCM = _crypto()
        plaintext = json.dumps(
            self._validate_keys(keys),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(self._key).encrypt(nonce, plaintext, _VAULT_AAD)
        envelope = {
            "format": _VAULT_FORMAT,
            "version": _VAULT_VERSION,
            "kdf": {"name": "scrypt", "n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P},
            "salt": _b64encode(self._salt),
            "nonce": _b64encode(nonce),
            "ciphertext": _b64encode(ciphertext),
        }
        payload = (json.dumps(envelope, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _atomic_write_private(self.path, payload)

    def unlock(self, passphrase: str) -> dict[str, str]:
        if not isinstance(passphrase, str) or not passphrase or len(passphrase.encode("utf-8")) > 4096:
            raise CredentialVaultError("enter a vault passphrase")
        AESGCM = _crypto()
        if self._looks_encrypted(self.path):
            key, salt, keys = self._read_envelope(passphrase)
            self._key, self._salt, self._keys = key, salt, keys
            return dict(self._keys)

        if len(passphrase) < _MIN_PASSPHRASE_LENGTH:
            raise CredentialVaultError(
                f"new vault passphrases must be at least {_MIN_PASSPHRASE_LENGTH} characters"
            )

        source: Path | None = None
        legacy_keys: dict[str, str] = {}
        if self.path.is_file():
            source = self.path
            legacy_keys = _read_legacy_json(source)
        elif self.legacy_path is not None and self.legacy_path.is_file():
            source = self.legacy_path
            legacy_keys = _read_legacy_json(source)

        salt = secrets.token_bytes(16)
        key = _derive_key(passphrase, salt)
        self._key, self._salt, self._keys = key, salt, dict(legacy_keys)
        try:
            self._write_envelope(self._keys)
        except Exception:
            self._key, self._salt, self._keys = None, None, {}
            raise

        # Remove the prior plaintext file only after the encrypted vault has
        # been written successfully. Unlinking cannot guarantee physical-media
        # erasure, but avoids leaving an ordinary readable credential file.
        if source is not None and source != self.path:
            try:
                source.unlink()
            except OSError:
                # Encrypted vault is already usable; status warns about legacy data.
                pass
        return dict(self._keys)

    def save_keys(self, keys: dict[str, str]) -> dict[str, str]:
        if not self.is_unlocked:
            raise CredentialVaultError("unlock the encrypted credential vault first")
        safe_keys = self._validate_keys(keys)
        self._write_envelope(safe_keys)
        self._keys = safe_keys
        return dict(self._keys)

    def keys(self) -> dict[str, str]:
        if not self.is_unlocked:
            return {}
        return dict(self._keys)

    def lock(self) -> None:
        self._keys.clear()
        self._key = None
        self._salt = None
