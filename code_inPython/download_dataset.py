import os, time, requests, urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin

DATA_DIR = "vowel_data_raw"
os.makedirs(DATA_DIR, exist_ok=True)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
session = requests.Session()
session.mount("http://", requests.adapters.HTTPAdapter())
session.mount("https://", requests.adapters.HTTPAdapter())

def complete_url(base_url, href):
    href = (href or "").strip()
    if href.startswith(("http://", "https://")):
        return href
    if href.startswith("//"):
        return "https:" + href
    if "utdallas.edu" in href:
        return "https://" + href.lstrip("/")
    if href.startswith(("assmann/KIDVOW/", "/assmann/KIDVOW/")):
        return "https://www.utdallas.edu/~" + href.lstrip("/")
    return urljoin(base_url, href)

def fetch_audio(url, retries=3, timeout=30):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*"
    }
    for _ in range(retries):
        try:
            r = session.get(url, timeout=timeout, verify=False, headers=headers)
            if r.ok and r.content:
                return r.content, r.headers
        except Exception:
            time.sleep(0.5)
    return b"", {}

def is_wav_file(content, headers):
    ct = (headers.get("Content-Type") or "").lower()
    if "audio" in ct or "wav" in ct:
        return True
    if content[:4] == b"RIFF" and b"WAVE" in content[:16]:
        return True
    if b"<html" in content[:200].lower():
        return False
    return False

def process_dataset_page(page_url):
    print(f"\n Fetching from: {page_url}")
    try:
        page = session.get(page_url, verify=False, timeout=40)
        page.raise_for_status()
    except Exception as e:
        print(f" Failed to load: {e}")
        return

    soup = BeautifulSoup(page.text, "html.parser")
    anchors = soup.find_all("a")
    current_word = "unknown"

    saved = skipped = existed = 0

    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href.endswith(".wav"):
            continue

        prev = a.find_previous(["b", "strong"])
        if prev:
            current_word = prev.get_text(strip=True).lower()

        word_dir = os.path.join(DATA_DIR, current_word)
        os.makedirs(word_dir, exist_ok=True)

        file_url = complete_url(page_url, href)
        filename = os.path.basename(file_url)
        local_path = os.path.join(word_dir, filename)

        if os.path.exists(local_path):
            existed += 1
            continue

        content, headers = fetch_audio(file_url)
        if not is_wav_file(content, headers):
            skipped += 1
            print(f" Skipped (invalid): {current_word}/{filename}")
            continue

        with open(local_path, "wb") as f:
            f.write(content)
        saved += 1
        print(f" Saved: {current_word}/{filename}")

    print(f"\nDone with page:\n Saved: {saved} |  Skipped: {skipped} |  Already existed: {existed}")

pages = [
    "https://personal.utdallas.edu/~assmann/KIDVOW/age7.html",
    "https://personal.utdallas.edu/~assmann/KIDVOW/admales.html",
    "https://personal.utdallas.edu/~assmann/KIDVOW/adfem.html",
    "https://personal.utdallas.edu/~assmann/KIDVOW/age5.html",
    "https://personal.utdallas.edu/~assmann/KIDVOW/age3.html",
]

for url in pages:
    process_dataset_page(url)
