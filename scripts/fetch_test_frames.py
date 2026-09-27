"""Добор тестовых кадров с Wikimedia (gsroffset — продолжение выдачи поиска)."""
import json
import re
import subprocess
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test_frames"
OUT.mkdir(parents=True, exist_ok=True)
UA = "SII-LR02-hammer-lab/1.0 (student project)"
API = "https://commons.wikimedia.org/w/api.php"

params = urllib.parse.urlencode({
    "action": "query", "generator": "search",
    "gsrsearch": "filetype:bitmap hammer head tool",
    "gsrnamespace": 6, "gsrlimit": 40, "gsroffset": 60,
    "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime",
    "iiurlwidth": 960, "format": "json",
})
r = subprocess.run(["curl", "-s", "--max-time", "30", "-A", UA, f"{API}?{params}"],
                   capture_output=True, text=True)
data = json.loads(r.stdout)
pages = sorted(((data.get("query") or {}).get("pages") or {}).items(),
               key=lambda kv: kv[1].get("index", 999))
n = 0
for _, page in pages:
    if n >= 8:
        break
    ii = (page.get("imageinfo") or [{}])[0]
    if ii.get("mime") != "image/jpeg" or ii.get("width", 0) < 640:
        continue
    url = (ii.get("thumburl") or "").split("?")[0]
    if not url:
        continue
    meta = ii.get("extmetadata") or {}
    lic = re.sub(r"<[^>]+>", "", (meta.get("LicenseShortName") or {}).get("value", ""))
    n += 1
    out = OUT / f"test_{n:02d}.jpg"
    code = subprocess.run(["curl", "-s", "--max-time", "40", "-A", UA, "-o", str(out), "-w", "%{http_code}", url],
                          capture_output=True, text=True).stdout
    ok = code == "200" and out.stat().st_size > 15000
    print(out.name, code, "OK" if ok else "BAD", lic)
    if not ok:
        out.unlink(missing_ok=True)
        n -= 1
    time.sleep(0.4)
