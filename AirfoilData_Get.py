import requests
import os
import time
import random
from urllib.parse import urljoin
import re


def download_all_airfoils(base_url, output_dir="all_airfoils"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        response = session.get(base_url, timeout=30)
        response.raise_for_status()

        dat_links = re.findall(r'href="(coord/[^"]+\.dat)"', response.text, re.IGNORECASE)
        unique_links = list(set(dat_links))

        downloaded_count = 0
        error_count = 0
        skipped_count = 0

        random.shuffle(unique_links)

        for i, rel_path in enumerate(unique_links):
            file_url = urljoin(base_url, rel_path)
            filename = os.path.basename(rel_path)
            local_path = os.path.join(output_dir, filename)

            if os.path.exists(local_path):
                skipped_count += 1
                continue

            try:
                time.sleep(random.uniform(1, 3))
                file_response = session.get(file_url, timeout=30)
                file_response.raise_for_status()

                content = file_response.content
                if len(content) < 50:
                    error_count += 1
                    continue

                with open(local_path, 'wb') as f:
                    f.write(content)

                downloaded_count += 1

            except Exception:
                error_count += 1

        print(f"Downloaded: {downloaded_count}, Skipped: {skipped_count}, Errors: {error_count}")

    except Exception as e:
        print(f"Error: {str(e)}")


def validate_downloaded_files(directory="all_airfoils"):
    if not os.path.exists(directory):
        return

    dat_files = [f for f in os.listdir(directory) if f.endswith('.dat')]

    size_stats = {}
    for filename in dat_files:
        try:
            size = os.path.getsize(os.path.join(directory, filename))
            size_range = f"{(size // 500) * 500}-{(size // 500) * 500 + 499}"
            size_stats[size_range] = size_stats.get(size_range, 0) + 1
        except OSError:
            pass

    sample_files = random.sample(dat_files, min(10, len(dat_files)))
    for filename in sample_files:
        file_path = os.path.join(directory, filename)
        try:
            size = os.path.getsize(file_path)
            print(f"{filename} ({size})")
        except OSError:
            print(f"{filename} (error)")


if __name__ == "__main__":
    base_url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
    download_all_airfoils(base_url)
    validate_downloaded_files()
