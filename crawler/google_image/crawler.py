import hashlib
import re
import time
import requests
from pathlib import Path
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from .models import CrawlConfig


class GoogleImageCrawler:
    BASE_URL = "https://www.google.com/search"
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}

    _COLOR_MAP = {
        "red": "ic:specific,isc:red", "orange": "ic:specific,isc:orange",
        "yellow": "ic:specific,isc:yellow", "green": "ic:specific,isc:green",
        "teal": "ic:specific,isc:teal", "blue": "ic:specific,isc:blue",
        "purple": "ic:specific,isc:purple", "pink": "ic:specific,isc:pink",
        "white": "ic:specific,isc:white", "gray": "ic:specific,isc:gray",
        "black": "ic:specific,isc:black", "brown": "ic:specific,isc:brown",
    }
    _COLOR_TYPE_MAP = {"color": "ic:color", "gray": "ic:gray", "transparent": "ic:trans"}
    _SIZE_MAP = {
        "large": "isz:l", "medium": "isz:m", "icon": "isz:i",
        "400x300": "isz:ex,iszw:400,iszh:300", "640x480": "isz:ex,iszw:640,iszh:480",
        "800x600": "isz:ex,iszw:800,iszh:600", "1024x768": "isz:ex,iszw:1024,iszh:768",
        "2mp": "isz:qsvga", "4mp": "isz:vga", "6mp": "isz:svga",
        "8mp": "isz:xga", "10mp": "isz:2mp", "12mp": "isz:4mp",
        "15mp": "isz:6mp", "20mp": "isz:8mp", "40mp": "isz:10mp", "70mp": "isz:12mp",
    }
    _IMAGE_TYPE_MAP = {
        "clipart": "itp:clipart", "lineart": "itp:lineart", "gif": "itp:animated",
        "face": "itp:face", "photo": "itp:photo", "animated": "itp:animated",
    }
    _USAGE_MAP = {
        "cc_publicdomain": "sur:fmc", "cc_attribute": "sur:fc",
        "cc_sharealike": "sur:fm", "cc_noncommercial": "sur:f",
        "cc_nonderived": "sur:fmc",
    }
    _ASPECT_MAP = {
        "tall": "iar:t", "square": "iar:s", "wide": "iar:w", "panoramic": "iar:xw",
    }
    _TIME_MAP = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})
        self._output_path = Path(config.output_dir)
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._downloaded = 0
        self._existing_hashes: set[str] = self._load_existing_hashes()
        print(f"Found {len(self._existing_hashes)} existing image(s) in '{config.output_dir}', duplicates will be skipped.")

    def _load_existing_hashes(self) -> set[str]:
        hashes: set[str] = set()
        for file in self._output_path.iterdir():
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                hashes.add(file.stem)
        return hashes

    @staticmethod
    def _hash_content(data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=20).hexdigest()

    def _build_tbs(self) -> str:
        f = self.config.image_filter
        parts = []
        if f.size and f.size in self._SIZE_MAP:
            parts.append(self._SIZE_MAP[f.size])
        if f.color and f.color in self._COLOR_MAP:
            parts.append(self._COLOR_MAP[f.color])
        elif f.color_type and f.color_type in self._COLOR_TYPE_MAP:
            parts.append(self._COLOR_TYPE_MAP[f.color_type])
        if f.image_type and f.image_type in self._IMAGE_TYPE_MAP:
            parts.append(self._IMAGE_TYPE_MAP[f.image_type])
        if f.usage_rights and f.usage_rights in self._USAGE_MAP:
            parts.append(self._USAGE_MAP[f.usage_rights])
        if f.aspect_ratio and f.aspect_ratio in self._ASPECT_MAP:
            parts.append(self._ASPECT_MAP[f.aspect_ratio])
        if f.time_range and f.time_range in self._TIME_MAP:
            parts.append(self._TIME_MAP[f.time_range])
        return ",".join(parts)

    def _build_query(self, keyword: str) -> str:
        if self.config.image_filter.site:
            keyword = f"site:{self.config.image_filter.site} {keyword}"
        return keyword

    def _build_url(self, start: int = 0) -> str:
        tbs = self._build_tbs()
        params = {
            "q": self._build_query(self.config.keyword),
            "tbm": "isch",
            "hl": "en",
            "start": start,
        }
        safe = self.config.image_filter.safe_search
        if safe != "off":
            params["safe"] = "active" if safe == "high" else "images"
        if tbs:
            params["tbs"] = tbs
        return f"{self.BASE_URL}?{urlencode(params)}"

    def _extract_image_urls(self, html: str) -> list[str]:
        urls = re.findall(r'"(https?://[^"]+\.(?:jpg|jpeg|png|gif|bmp|webp))"', html)
        soup = BeautifulSoup(html, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if src.startswith("http") and not src.startswith("https://encrypted-tbn"):
                urls.append(src)
        seen: set[str] = set()
        unique = []
        for u in urls:
            if u not in seen and "gstatic" not in u and "google" not in u:
                seen.add(u)
                unique.append(u)
        return unique

    def _get_extension(self, url: str, content_type: str = "") -> str:
        for ext in self.SUPPORTED_EXTENSIONS:
            if url.lower().endswith(ext):
                return ext
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
        if "gif" in content_type:
            return ".gif"
        if "webp" in content_type:
            return ".webp"
        return ".jpg"

    def _download_image(self, url: str) -> bool:
        try:
            resp = self.session.get(url, timeout=self.config.timeout)
            if resp.status_code != 200:
                return False
            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type and not any(e in content_type for e in ["jpeg", "png", "gif", "webp"]):
                return False

            content = resp.content
            content_hash = self._hash_content(content)

            if content_hash in self._existing_hashes:
                print(f"  Skipped (duplicate): {url[:60]}...")
                return False

            ext = self._get_extension(url, content_type)
            filename = self._output_path / f"{content_hash}{ext}"
            filename.write_bytes(content)

            self._existing_hashes.add(content_hash)
            self._downloaded += 1
            print(f"[{self._downloaded}/{self.config.limit}] Saved: {filename.name}")
            return True
        except Exception as e:
            print(f"  Failed: {url[:60]}... -> {e}")
            return False

    def crawl(self) -> int:
        print(f"Starting crawl: '{self.config.keyword}' | limit={self.config.limit}")
        start = 0
        collected_urls: list[str] = []

        while len(collected_urls) < self.config.limit * 3:
            url = self._build_url(start)
            try:
                resp = self.session.get(url, timeout=self.config.timeout)
                urls = self._extract_image_urls(resp.text)
                if not urls:
                    break
                for u in urls:
                    if u not in collected_urls:
                        collected_urls.append(u)
                start += 20
                time.sleep(self.config.request_delay)
            except Exception as e:
                print(f"Error fetching search page: {e}")
                break

        print(f"Found {len(collected_urls)} candidate URLs, downloading up to {self.config.limit}...")
        for url in collected_urls:
            if self._downloaded >= self.config.limit:
                break
            self._download_image(url)
            time.sleep(self.config.request_delay)

        print(f"\nDone. Downloaded {self._downloaded} images to '{self.config.output_dir}'")
        return self._downloaded
