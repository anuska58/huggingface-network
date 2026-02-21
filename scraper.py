    """
    Scrapes model metadata and images from Hugging Face, saving them to a CSV file and an images directory.
    The script uses multithreading to speed up the image downloading process and includes error handling for network issues and missing data.
    Progress is saved to a JSON file to allow resuming the scraping process in case of interruptions.
    """

import os
import re
import json
import time
import logging
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download


base_url = "https://huggingface.co/api/models"
csv_file = "model_metadata.csv"
progress_file = "progress.json"
image_dir = "images"
max_workers = 10
requests_timeout = 15
batch_size = 1000

os.makedirs(image_dir, exist_ok=True)


logging.basicConfig(
    filename="scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

os.makedirs(image_dir, exist_ok=True)

def load_progress():
    """Loads the last progress from the progress file, returning the last created_at timestamp if available."""
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get("last_created_at")
    return None

def save_progress(timestamp):
    """Saves the last created_at timestamp to the progress file."""
    with open(progress_file, "w", encoding="utf-8") as file:
        json.dump({"last_created_at": timestamp}, file)

def create_session():
    """Creates a requests session with retry logic to handle transient network issues."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=5)
    session.mount("http://", adapter)
    return session

def fetch_new_models(session, last_created_at):
    """Fetches new models from the Hugging Face API created after the last_created_at timestamp."""
    params = {
        "sort": "created_at",
        "direction": -1,
        "limit": batch_size
    }

    response = session.get(base_url, params=params, timeout=requests_timeout)
    response.raise_for_status()
    models = response.json()

    new_models = []

    for model in models:
        created_at = model.get("created_at")
        if not created_at:
            continue
        if last_created_at and created_at <= last_created_at:
            break
        new_models.append(model)

    return new_models
        