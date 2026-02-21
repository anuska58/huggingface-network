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
    params = {"sort": "created_at", "direction": -1, "limit": batch_size}

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


def download_images(session, model_card, creator, model_name):
    """Downloads images from the model card, saving them to the appropriate directory
    and returning the count of successfully downloaded images."""
    image_urls = re.findall(r"!\[.*?\]\((https?://.*?)\)", model_card)

    creator_folder = os.path.join(image_dir, creator)
    os.makedirs(creator_folder, exist_ok=True)

    image_count = 0

    for index, image_url in enumerate(image_urls):
        try:
            if not image_url.startswith("http"):
                continue

            allowed_extensions = {"png", "jpg", "jpeg", "gif", "webp"}

            extension = image_url.split("?")[0].split(".")[-1].lower()

            if extension not in allowed_extensions:
                extension = "jpg"

            filename = f"{model_name}_{index + 1}.{extension}"
            filepath = os.path.join(creator_folder, filename)

            if os.path.exists(filepath):
                image_count += 1
                continue

            response = session.get(image_url, timeout=requests_timeout)

            if response.status_code == 200:
                with open(filepath, "wb") as file:
                    file.write(response.content)
                image_count += 1

        except Exception as error:
            logging.warning("Image download failed: %s", error)

    return image_count


def process_model(session, model):

    try:
        model_id = model["modelId"]
        creator = model_id.split("/")[0]
        model_name = model_id.split("/")[1]
        url = f"https://huggingface.co/{model_id}"
        task = model.get("pipeline_tag")
        created_at = model.get("created_at")

        if not task:
            return None

        try:
            readme_path = hf_hub_download(
                repo_id=model_id,
                filename="README.md",
            )
            with open(readme_path, "r", encoding="utf-8") as file:
                model_card = file.read()

        except Exception:
            model_card = ""

        image_count = download_images(
            session,
            model_card,
            creator,
            model_name,
        )

        return {
            "task": task,
            "creator": creator,
            "model_name": model_name,
            "url": url,
            "model_card": model_card,
            "image_count": image_count,
            "created_at": created_at,
        }
    except Exception as error:
        logging.error("Model processing failed: %s", error)
        return None


def main():
    """Main function to orchestrate the scraping process, including fetching new models, processing them, and saving results."""
    session = create_session()
    last_created_at = load_progress()

    models = fetch_new_models(session, last_created_at)

    if not models:
        print("No new models found.")
        return

    print(f"Found {len(models)} new models. Processing...")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [executor.submit(process_model, session, model) for model in models]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                results.append(result)

        if not results:
            print("No valid models processed.")
            return

        df = pd.DataFrame(results)

        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False, encoding="utf-8")
        else:
            df.to_csv(csv_file, mode="a", header=False, index=False, encoding="utf-8")

        new_timestamp = results[0]["created_at"]
        save_progress(new_timestamp)

        print(f"Processed {len(results)} models. Progress saved. Batch completed.")


if __name__ == "__main__":
    main()
