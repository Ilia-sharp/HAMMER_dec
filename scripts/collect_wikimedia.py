"""
ЛР2, вариант 16 (hammer): сбор изображений с Wikimedia Commons API.
Использует curl (Wikimedia блокирует TLS-fingerprint python-клиентов).
Скачивает фото молотков + сохраняет licenses.csv (требование лабораторной).
Запуск: python scripts/collect_wikimedia.py
"""
import csv
import json
import re
import subprocess
import time
import urllib.parse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
UA = "SII-LR02-hammer-lab/1.0 (student project; contact: ivanov@example.edu)"
TARGET = 70  # скачиваем с запасом, потом QA-отбор до 55
QUERIES = ["hammer", "claw hammer", "ball-peen hammer", "sledgehammer", "carpenter hammer tool"]


def curl(url: str, out: str | None = None, timeout: int = 40) -> str | int:
    """Возвращает текст ответа или код HTTP, если out не задан."""
    cmd = ["curl", "-s", "--max-time", str(timeout), "-A", UA]
    if out:
        cmd += ["-o", out, "-w", "%{http_code}", url]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return int(r.stdout.strip() or 0)
    r = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=timeout + 10)
    return r.stdout


def clean_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def main():
    rows = []
    seen = set()
    n = 0
    for q in QUERIES:
        if n >= TARGET:
            break
        params = urllib.parse.urlencode({
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {q}",
            "gsrnamespace": 6,
            "gsrlimit": 40,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime",
            "iiurlwidth": 960,
            "format": "json",
        })
        try:
            data = json.loads(curl(f"{API}?{params}"))
        except Exception as e:
            print(f"  API error ({q}): {e}")
            time.sleep(3)
            continue
        pages = (data.get("query") or {}).get("pages") or {}
        for _, page in sorted(pages.items(), key=lambda kv: kv[1].get("index", 999)):
            if n >= TARGET:
                break
            ii = (page.get("imageinfo") or [{}])[0]
            mime = ii.get("mime", "")
            w, h = ii.get("width", 0), ii.get("height", 0)
            title = page.get("title", "")
            if mime not in ("image/jpeg", "image/png") or w < 640 or title.lower() in seen:
                continue
            url = (ii.get("thumburl") or ii.get("url") or "").split("?")[0]
            if not url:
                continue
            meta = ii.get("extmetadata") or {}
            artist = clean_html((meta.get("Artist") or {}).get("value", ""))[:80]
            lic = clean_html((meta.get("LicenseShortName") or {}).get("value", ""))
            ext = ".png" if mime == "image/png" else ".jpg"
            fname = f"wiki_{n + 1:03d}{ext}"
            code = curl(url, out=str(RAW / fname), timeout=60)
            if code == 200 and (RAW / fname).stat().st_size > 15000:
                n += 1
                seen.add(title.lower())
                rows.append([fname, title, artist, lic, ii.get("descriptionurl", "")])
                print(f"[{n:02d}] {fname}  <- {title[:55]}  ({lic})")
            else:
                (RAW / fname).unlink(missing_ok=True)
                print(f"  skip {title[:50]} (http {code})")
            time.sleep(0.4)

    with open(RAW / "licenses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "wikimedia_title", "artist", "license", "source_url"])
        w.writerows(rows)
    print(f"\nDownloaded: {n} images -> {RAW}")
    print(f"licenses.csv rows: {len(rows)}")


if __name__ == "__main__":
    main()
