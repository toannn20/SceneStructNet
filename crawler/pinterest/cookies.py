"""
Utility to normalise cookies.txt to Netscape format.

Supported input formats:
  1. Netscape / Mozilla (standard gallery-dl format)       — pass through unchanged
  2. JSON array  [ {name, value, domain, ...}, ... ]       — Chrome EditThisCookie / Cookie-Editor
  3. JSON object { "url": "...", "cookies": [ ... ] }      — Cookie-Editor export (what the user has)
"""

import json
import math
import tempfile
from pathlib import Path


_NETSCAPE_HEADER = "# Netscape HTTP Cookie File"


def _is_netscape(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("#") or ("\t" in stripped.split("\n")[0])


def _json_cookie_to_netscape_line(c: dict) -> str:
    domain = c.get("domain", "")
    include_subdomains = "FALSE" if c.get("hostOnly", False) else "TRUE"
    path = c.get("path", "/")
    secure = "TRUE" if c.get("secure", False) else "FALSE"
    expiry = c.get("expirationDate", 0)
    expiry_int = 0 if (expiry is None or math.isnan(float(expiry))) else int(expiry)
    name = c.get("name", "")
    value = c.get("value", "")
    return "\t".join([domain, include_subdomains, path, secure, str(expiry_int), name, value])


def _json_to_netscape(text: str) -> str:
    data = json.loads(text)
    if isinstance(data, dict) and "cookies" in data:
        data = data["cookies"]
    if not isinstance(data, list):
        raise ValueError("Unrecognised JSON cookies format")
    lines = [_NETSCAPE_HEADER]
    for c in data:
        lines.append(_json_cookie_to_netscape_line(c))
    return "\n".join(lines) + "\n"


def resolve_cookies_file(cookies_file: str) -> str:
    """
    Read cookies_file, convert to Netscape if needed, write to a temp file,
    and return the path to a guaranteed-Netscape cookies file.
    The caller is responsible for deleting the temp file when done.
    Returns the original path unchanged if it's already Netscape format.
    """
    path = Path(cookies_file).expanduser().resolve()
    text = path.read_text(encoding="utf-8", errors="replace").strip()

    if _is_netscape(text):
        return str(path)

    try:
        netscape_text = _json_to_netscape(text)
    except Exception as exc:
        raise ValueError(
            f"cookies.txt at '{path}' is neither Netscape nor recognised JSON format.\n"
            f"Error: {exc}\n"
            "Export your cookies using 'Get cookies.txt LOCALLY' (Chrome) or "
            "'Export Cookies' (Firefox) for Netscape format."
        ) from exc

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_cookies.txt", delete=False, encoding="utf-8"
    )
    tmp.write(netscape_text)
    tmp.close()
    return tmp.name
