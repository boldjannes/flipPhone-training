"""
Fetch the latest dataset from the FlipPhone server as CSV.

Usage:
    python fetch_data.py --url https://your-server.com --key fp_yourAdminKey
"""

import argparse
import os
import sys

import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch(server_url: str, api_key: str, min_confidence: float | None = None) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    url = f"{server_url.rstrip('/')}/admin/api/export/csv"
    params = {}
    if min_confidence is not None:
        params["min_confidence"] = min_confidence

    print(f"Fetching dataset from {server_url} …")
    resp = requests.get(url, headers={"X-API-Key": api_key}, params=params, timeout=30)
    resp.raise_for_status()

    out_path = os.path.join(DATA_DIR, "dataset.csv")
    with open(out_path, "wb") as f:
        f.write(resp.content)

    line_count = resp.text.count("\n") - 1  # subtract header
    print(f"Saved {line_count} sample rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch FlipPhone dataset")
    parser.add_argument("--url", required=True, help="Server URL (e.g. https://flip.example.com)")
    parser.add_argument("--key", required=True, help="Admin API key")
    args = parser.parse_args()
    try:
        fetch(args.url, args.key)
    except requests.HTTPError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
