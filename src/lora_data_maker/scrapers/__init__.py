from .base import ImageResult, Scraper, ScraperError
from .direct import DirectURLScraper
from .google import GoogleImagesScraper
from .manager import ScraperManager

__all__ = [
    "DirectURLScraper",
    "GoogleImagesScraper",
    "ImageResult",
    "Scraper",
    "ScraperError",
    "ScraperManager",
]
