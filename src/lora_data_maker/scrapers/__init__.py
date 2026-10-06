from .base import ImageResult, Scraper, ScraperError
from .chromium import ChromiumGoogleImagesScraper
from .direct import DirectURLScraper
from .duckduckgo import DuckDuckGoImagesScraper
from .google import GoogleImagesScraper
from .manager import ScraperManager

__all__ = [
    "ChromiumGoogleImagesScraper",
    "DirectURLScraper",
    "DuckDuckGoImagesScraper",
    "GoogleImagesScraper",
    "ImageResult",
    "Scraper",
    "ScraperError",
    "ScraperManager",
]
