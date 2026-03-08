import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_MAIN = "https://diatoms.org"

IMG_EXT_RE = re.compile(r"\.(jpg|jpeg|png|gif|tif|tiff)(?:\?|#|$)", re.IGNORECASE)


def slugify_species(name):
    # Convert species name to diatoms.org format
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def sanitize_dirname(name):
    # Make species name safe
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    return name


def get_html(session, url, timeout=30):
    r = session.get(url, timeout=timeout, allow_redirects=True)
    return r.status_code, r.text


def to_fallback(url):
    if url.startswith(BASE_MAIN):
        return url.replace(BASE_MAIN, BASE_FALLBACK, 1)
    return url


def resolve_species_base(session, species):
    slug = slugify_species(species)

    candidates = [
        f"{BASE_MAIN}/species/{slug}",
        f"{BASE_MAIN}/species/{slug.replace('_','-')}",
        f"{BASE_FALLBACK}/species/{slug}",
        f"{BASE_FALLBACK}/species/{slug.replace('_','-')}",
    ]

    for url in candidates:
        code, _ = get_html(session, url)
        if code == 200:
            return url

    return None


def extract_detail_links(html, base_url):
    # Extract /images/<id> page links
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        if re.search(r"/images/\d+", a["href"]):
            links.append(urljoin(base_url, a["href"]))

    return list(dict.fromkeys(links))


def extract_image_urls(html, page_url):
    # Extract image URLs
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        urls.append(urljoin(page_url, og["content"]))

    for img in soup.find_all("img", src=True):
        if IMG_EXT_RE.search(img["src"]):
            urls.append(urljoin(page_url, img["src"]))

    for a in soup.find_all("a", href=True):
        if IMG_EXT_RE.search(a["href"]):
            urls.append(urljoin(page_url, a["href"]))

    return list(dict.fromkeys(urls))


def download_file(session, url, out_path):
    # Download image file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)


def crawl_species(session, species, out_root, sleep_s=0.5):
    # Download images for a single species
    base_url = resolve_species_base(session, species)
    if base_url is None:
        return {"species": species, "saved": 0}

    list_url = base_url.rstrip("/") + "/images"
    code, html = get_html(session, list_url)

    if code != 200:
        list_url = to_fallback(list_url)
        code, html = get_html(session, list_url)
        if code != 200:
            return {"species": species, "saved": 0}

    detail_links = extract_detail_links(html, list_url)

    out_dir = os.path.join(out_root, sanitize_dirname(species))
    os.makedirs(out_dir, exist_ok=True)

    saved = 0

    for detail_url in detail_links:
        code, detail_html = get_html(session, detail_url)

        if code != 200:
            detail_url = to_fallback(detail_url)
            code, detail_html = get_html(session, detail_url)
            if code != 200:
                continue

        image_urls = extract_image_urls(detail_html, detail_url)

        for img_url in image_urls:
            try:
                ext = os.path.splitext(img_url.split("?")[0])[1].lower() or ".jpg"
                filename = f"{saved+1:04d}{ext}"
                path = os.path.join(out_dir, filename)

                download_file(session, img_url, path)
                saved += 1
                break

            except Exception:
                continue

        time.sleep(sleep_s)

    return {"species": species, "saved": saved}


def load_species_list(path):
    # Read species list
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():

    species_list = load_species_list("species.txt")

    out_root = "diatoms_org_images"
    os.makedirs(out_root, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (diatoms-crawler)"
    })

    results = []

    for sp in tqdm(species_list):
        results.append(crawl_species(session, sp, out_root))

    success = [r for r in results if r["saved"] > 0]
    failed = [r for r in results if r["saved"] == 0]

    print("Saved root:", os.path.abspath(out_root))
    print("Success:", len(success))
    print("Failed:", len(failed))


if __name__ == "__main__":
    main()