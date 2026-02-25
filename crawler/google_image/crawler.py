import asyncio
import hashlib
import re
from pathlib import Path
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import CrawlConfig


class RateLimitedError(Exception):
    """Raised when a 429 or 5xx response is received."""

    def __init__(self, status_code: int, retry_after: float | None = None):
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"HTTP {status_code}")


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
        self._output_path = Path(config.output_dir)
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._downloaded = 0
        self._existing_hashes: set[str] = self._load_existing_hashes()
        self._lock = asyncio.Lock()
        self._limit_reached = asyncio.Event()
        print(
            f"Found {len(self._existing_hashes)} existing image(s) "
            f"in '{config.output_dir}', duplicates will be skipped."
        )

    # ------------------------------------------------------------------
    # Helpers (unchanged logic)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Retry decorator factory
    # ------------------------------------------------------------------

    def _make_retry(self):
        """Build a tenacity retry decorator from the current config."""
        return retry(
            retry=retry_if_exception_type(RateLimitedError),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=1,
                max=30,
                exp_base=self.config.backoff_base,
            ),
            reraise=True,
        )

    @staticmethod
    def _check_response(resp: httpx.Response) -> None:
        """Raise RateLimitedError on 429 / 5xx so tenacity can retry."""
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else None
            if wait:
                print(f"  ⚠ Rate-limited (429). Retry-After: {wait}s")
            raise RateLimitedError(429, retry_after=wait)
        if resp.status_code >= 500:
            raise RateLimitedError(resp.status_code)

    # ------------------------------------------------------------------
    # Async network methods
    # ------------------------------------------------------------------

    async def _fetch_search_page(
        self, client: httpx.AsyncClient, start: int
    ) -> str | None:
        """Fetch a single Google Image search page with retry."""
        url = self._build_url(start)

        @self._make_retry()
        async def _inner():
            resp = await client.get(url)
            self._check_response(resp)
            resp.raise_for_status()
            return resp.text

        try:
            return await _inner()
        except Exception as exc:
            print(f"Error fetching search page (start={start}): {exc}")
            return None

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
    ) -> bool:
        """Download a single image, bounded by semaphore, with retry."""
        if self._limit_reached.is_set():
            return False

        async with semaphore:
            if self._limit_reached.is_set():
                return False

            @self._make_retry()
            async def _inner():
                resp = await client.get(url)
                self._check_response(resp)
                if resp.status_code != 200:
                    return False

                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type and not any(
                    e in content_type for e in ["jpeg", "png", "gif", "webp"]
                ):
                    return False

                content = resp.content
                content_hash = self._hash_content(content)

                async with self._lock:
                    if content_hash in self._existing_hashes:
                        print(f"  Skipped (duplicate): {url[:60]}...")
                        return False

                    if self._downloaded >= self.config.limit:
                        self._limit_reached.set()
                        return False

                    ext = self._get_extension(url, content_type)
                    filename = self._output_path / f"{content_hash}{ext}"
                    filename.write_bytes(content)

                    self._existing_hashes.add(content_hash)
                    self._downloaded += 1
                    print(f"[{self._downloaded}/{self.config.limit}] Saved: {filename.name}")

                    if self._downloaded >= self.config.limit:
                        self._limit_reached.set()

                    return True

            try:
                return await _inner()
            except Exception as exc:
                print(f"  Failed: {url[:60]}... -> {exc}")
                return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def crawl(self) -> int:
        """Crawl Google Images asynchronously. Returns the number of
        images downloaded."""
        print(f"Starting crawl: '{self.config.keyword}' | limit={self.config.limit}")

        limits = httpx.Limits(
            max_connections=self.config.max_concurrency,
            max_keepalive_connections=self.config.max_concurrency,
        )
        timeout = httpx.Timeout(self.config.timeout, connect=self.config.timeout)

        async with httpx.AsyncClient(
            http2=True,
            limits=limits,
            timeout=timeout,
            headers={"User-Agent": self.config.user_agent},
            follow_redirects=True,
        ) as client:
            # Phase 1: collect candidate URLs (sequential pagination)
            collected_urls: list[str] = []
            start = 0

            while len(collected_urls) < self.config.limit * 3:
                html = await self._fetch_search_page(client, start)
                if html is None:
                    break
                urls = self._extract_image_urls(html)
                if not urls:
                    break
                for u in urls:
                    if u not in collected_urls:
                        collected_urls.append(u)
                start += 20
                await asyncio.sleep(self.config.request_delay)

            print(
                f"Found {len(collected_urls)} candidate URLs, "
                f"downloading up to {self.config.limit}..."
            )

            # Phase 2: download images concurrently
            semaphore = asyncio.Semaphore(self.config.max_concurrency)

            tasks = [
                asyncio.create_task(
                    self._download_image(client, semaphore, url)
                )
                for url in collected_urls
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

        print(f"\nDone. Downloaded {self._downloaded} images to '{self.config.output_dir}'")
        return self._downloaded
