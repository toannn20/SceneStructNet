import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse


def _keyword_to_folder(keyword: str) -> str:
    slug = keyword.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s]+", "_", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug or "images"


def _infer_label(url: str) -> str:
    parsed = urlparse(url)

    # Search URL: /search/pins/?q=mountain+landscape
    if "/search/" in parsed.path:
        qs = parse_qs(parsed.query)
        q = qs.get("q", [""])[0].strip()
        if q:
            return q

    parts = [
        p for p in parsed.path.strip("/").split("/")
        if p and p not in ("search", "pins", "pin")
    ]

    # /username/board-name/  → "username_board-name"
    # /username/             → "username"
    # /pin/123456/           → "pin_123456"
    return "_".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "pinterest")


@dataclass
class PinterestConfig:
    # ── Required ──────────────────────────────────────────────────────────────
    url: str                          # Pinterest board/pin/user/search URL

    # ── Download control ──────────────────────────────────────────────────────
    limit: int              = 50
    base_dir: str           = "downloads"
    output_dir: str         = ""

    # ── Pinterest-specific options ─────────────────────────────────────────────
    sections: bool          = True    # include pins from board sections
    videos: bool            = False   # download video pins
    story_pins: bool        = False   # extract files from story pins
    domain: str             = "auto"  # "auto" | "pinterest.com" | "pinterest.co.uk" etc.

    # ── Authentication ─────────────────────────────────────────────────────────
    cookies_file: Optional[str] = None

    # ── gallery-dl tuning ─────────────────────────────────────────────────────
    retries: int            = 3
    timeout: float          = 10.0
    sleep_request: float    = 0.5
    sleep_extractor: float  = 1.0
    filename_format: str    = "{id}.{extension}"

    # ── HTTP ──────────────────────────────────────────────────────────────────
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __post_init__(self) -> None:
        if not self.output_dir:
            slug = _keyword_to_folder(_infer_label(self.url))
            self.output_dir = str(Path(self.base_dir) / slug)
