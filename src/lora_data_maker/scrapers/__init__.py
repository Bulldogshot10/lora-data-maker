from .base import ImageResult, Scraper, ScraperError
from .direct import DirectURLScraper
from .duckduckgo import DuckDuckGoImagesScraper
from .google import GoogleImagesScraper
from .manager import ScraperManager

__all__ = [
    "DirectURLScraper",
    "DuckDuckGoImagesScraper",
    "GoogleImagesScraper",
    "ImageResult",
    "Scraper",
    "ScraperError",
    "ScraperManager",
]
