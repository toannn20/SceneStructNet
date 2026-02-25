import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageFilter:
    color: Optional[str] = None
    color_type: Optional[str] = None
    size: Optional[str] = None
    image_type: Optional[str] = None
    usage_rights: Optional[str] = None
    aspect_ratio: Optional[str] = None
    safe_search: str = "off"
    file_format: Optional[str] = None
    time_range: Optional[str] = None
    site: Optional[str] = None


def _keyword_to_folder(keyword: str) -> str:
    slug = keyword.strip().lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s]+', '_', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug or "images"


@dataclass
class CrawlConfig:
    keyword: str
    limit: int = 20
    base_dir: str = "downloads"
    output_dir: str = ""
    image_filter: ImageFilter = field(default_factory=ImageFilter)
    request_delay: float = 0.5
    timeout: int = 10
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = f"{self.base_dir}/{_keyword_to_folder(self.keyword)}"
