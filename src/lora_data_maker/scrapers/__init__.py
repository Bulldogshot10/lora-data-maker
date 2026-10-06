from .base import ImageResult, Scraper, ScraperError
from .chromium import ChromiumImagesScraper
from .direct import DirectURLScraper
from .duckduckgo import DuckDuckGoImagesScraper
from .google import GoogleImagesScraper
from .manager import ScraperManager

__all__ = [
    "ChromiumImagesScraper",
    "DirectURLScraper",
    "DuckDuckGoImagesScraper",
    "GoogleImagesScraper",
    "ImageResult",
    "Scraper",
    "ScraperError",
    "ScraperManager",
]
