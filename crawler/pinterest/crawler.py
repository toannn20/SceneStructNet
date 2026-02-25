import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from gallery_dl import config as gdl_config
from gallery_dl.job import DownloadJob

from .cookies import resolve_cookies_file
from .models import PinterestConfig


class _LimitReached(Exception):
    """Raised inside the job to abort once the download limit is hit."""


class _LimitedDownloadJob(DownloadJob):
    """DownloadJob subclass that stops after `max_downloads` files."""

    def __init__(self, url: str, max_downloads: int):
        super().__init__(url)
        self._max_downloads = max_downloads
        self._count = 0

    def handle_url(self, url: str, kwdict: dict) -> None:
        if self._count >= self._max_downloads:
            raise _LimitReached()
        super().handle_url(url, kwdict)
        self._count += 1

    def run(self) -> int:
        try:
            return super().run()
        except _LimitReached:
            return 0


class PinterestCrawler:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}

    def __init__(self, config: PinterestConfig):
        self.config = config
        self._output_path = Path(config.output_dir)
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._existing_hashes: set[str] = self._load_existing_hashes()
        self._downloaded = 0
        print(
            f"Found {len(self._existing_hashes)} existing image(s) "
            f"in '{config.output_dir}', duplicates will be skipped."
        )

    def _load_existing_hashes(self) -> set[str]:
        hashes: set[str] = set()
        for file in self._output_path.iterdir():
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                hashes.add(file.stem)
        return hashes

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.blake2b(digest_size=20)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _apply_gdl_config(self, tmp_dir: str, resolved_cookies: str | None) -> None:
        gdl_config.clear()

        gdl_config.set(("extractor",), "base-directory", tmp_dir)
        gdl_config.set(("extractor",), "retries",        self.config.retries)
        gdl_config.set(("extractor",), "timeout",        self.config.timeout)
        gdl_config.set(("extractor",), "sleep-request",  self.config.sleep_request)
        gdl_config.set(("extractor",), "user-agent",     self.config.user_agent)

        gdl_config.set(("extractor", "pinterest"), "directory", [])
        
        gdl_config.set(("extractor", "pinterest"), "filename", "{id}.{extension}")

        if resolved_cookies:
            gdl_config.set(("extractor", "pinterest"), "cookies", resolved_cookies)

        gdl_config.set(("extractor", "pinterest"), "domain",   self.config.domain)
        gdl_config.set(("extractor", "pinterest"), "sections", self.config.sections)

    def _process_downloads(self, tmp_dir: Path) -> int:
        all_files = [
            f for f in tmp_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        print(f"  Found {len(all_files)} file(s) in temp dir.")

        if not all_files:
            print("  ⚠ No files downloaded.")
            print("    • cookies.txt may be expired — re-export from your browser")
            print("    • Try increasing sleep_request to avoid rate limits")
            return 0

        saved = 0
        for file in sorted(all_files, key=lambda f: f.stat().st_mtime):
            if self._downloaded + saved >= self.config.limit:
                break

            content_hash = self._hash_file(file)

            if content_hash in self._existing_hashes:
                print(f"  Skipped (duplicate): {file.name}")
                continue

            dest = self._output_path / f"{content_hash}{file.suffix.lower()}"
            shutil.copy2(str(file), dest)
            self._existing_hashes.add(content_hash)
            saved += 1
            print(f"[{self._downloaded + saved}/{self.config.limit}] Saved: {dest.name}")

        return saved

    def crawl(self) -> int:
        if not self.config.cookies_file:
            print("⚠ Warning: no cookies_file set. Pinterest requires authentication.")

        print(f"Starting Pinterest crawl: '{self.config.url}' | limit={self.config.limit}")

        resolved_cookies: str | None = None
        tmp_cookies: str | None = None

        if self.config.cookies_file:
            resolved_cookies = resolve_cookies_file(self.config.cookies_file)
            if resolved_cookies != str(Path(self.config.cookies_file).expanduser().resolve()):
                tmp_cookies = resolved_cookies
                print("  ✓ Converted JSON cookies → Netscape format")
            else:
                print(f"  ✓ Cookies loaded: {resolved_cookies}")

        tmp_dir = Path(tempfile.mkdtemp(prefix="pinterest_gdl_"))
        print(f"  Temp dir: {tmp_dir}")

        try:
            self._apply_gdl_config(str(tmp_dir), resolved_cookies)

            job = _LimitedDownloadJob(self.config.url, self.config.limit * 2)
            job.run()

            print("\n  gallery-dl finished. Processing downloads...")
            self._downloaded += self._process_downloads(tmp_dir)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if tmp_cookies and os.path.exists(tmp_cookies):
                os.unlink(tmp_cookies)

        print(f"\nDone. Downloaded {self._downloaded} images to '{self.config.output_dir}'")
        return self._downloaded
