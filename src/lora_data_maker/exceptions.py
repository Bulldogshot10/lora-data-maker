class LoraDataMakerError(Exception):
    """Base exception for expected application errors."""


class ProjectError(LoraDataMakerError):
    """Raised for invalid or unusable project directories."""


class ConfigError(LoraDataMakerError):
    """Raised for invalid or unreadable project configuration."""
