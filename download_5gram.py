"""
Run this file to download the language model files from Dryad and place them in
language_model/pretrained_language_models/.

From the top-level directory of the repository (nejm-brain-to-text/), run:

    conda activate b2txt25
    python download_language_models.py

This will download languageModel.tar.gz and languageModel_5gram.tar.gz
from Dryad DOI 10.5061/dryad.x69p8czpq.
"""

import sys
import os
import urllib.request
import json


########################################################################################
#
# Helpers.
#
########################################################################################


def display_progress_bar(block_num, block_size, total_size, message=""):
    """Simple progress bar for urllib.urlretrieve."""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6 if total_size > 0 else 0
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()


########################################################################################
#
# Main function.
#
########################################################################################


def main():
    # Dryad DOI for the *language model* dataset
    DRYAD_DOI = "10.5061/dryad.x69p8czpq"

    # Make sure we're in the repo root
    repo_name = "nejm-brain-to-text"
    cwd = os.getcwd()
    assert cwd.endswith(
        repo_name
    ), f"Please run this script from the {repo_name} directory (instead of {cwd})"

    # Target directory for language models
    LM_DIR = os.path.join("language_model", "pretrained_language_models")
    lm_dirpath = os.path.abspath(LM_DIR)
    os.makedirs(lm_dirpath, exist_ok=True)

    print(f"Downloading language models into: {lm_dirpath}")

    # Dryad API root
    DRYAD_ROOT = "https://datadryad.org"
    urlified_doi = DRYAD_DOI.replace("/", "%2F")

    # 1) Get all versions for this DOI
    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    print(f"Querying Dryad versions from: {versions_url}")
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    # Take the *latest* version
    latest_version = versions_info["_embedded"]["stash:versions"][-1]
    files_url_path = latest_version["_links"]["stash:files"]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"

    print(f"Querying file list from: {files_url}")
    with urllib.request.urlopen(files_url) as response:
        files_info = json.loads(response.read().decode())

    file_infos = files_info["_embedded"]["stash:files"]

    # 2) Loop over files and download only the language model tarballs
    for file_info in file_infos:
        filename = file_info["path"]

        # Skip everything except languageModel*.*
        if not filename == "languageModel.tar.gz":
            print(f"Skipping {filename}")
            continue

        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"

        download_to_filepath = os.path.join(lm_dirpath, filename)
        print(f"\nDownloading {filename} from:\n  {download_url}")
        print(f"Saving to:\n  {download_to_filepath}\n")

        urllib.request.urlretrieve(
            download_url,
            download_to_filepath,
            reporthook=lambda *args, fname=filename: display_progress_bar(
                *args, message=f"Downloading {fname}"
            ),
        )
        sys.stdout.write("\n")

    print(f"\nDownload complete. See language models in {lm_dirpath}\n")


if __name__ == "__main__":
    main()
