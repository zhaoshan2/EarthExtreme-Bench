import os

# os.environ["EE_BENCH_DIR"] = "/home/EarthExtreme-Bench/data/zip-test"

from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
import zipfile

# from earthextremebench import EE_BENCH_DIRR

def decompress_zip_with_progress(zip_file_path, extract_to_folder=None):
    """Decompress a zip file with a progress bar and remove the symlink."""
    if extract_to_folder is None:
        extract_to_folder = zip_file_path.parent

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_names = zip_ref.namelist()
        total_files = len(file_names)

        # Initialize the progress bar with the total number of files
        with tqdm(total=total_files, unit="file", desc=f"Extracting {zip_file_path.name}") as pbar:
            for file in file_names:
                # Extract each file
                zip_ref.extract(file, extract_to_folder)
                # Update the progress bar
                pbar.update(1)

    # remove zip file
    zip_file_path.unlink()


def download_benchmark():
    EE_BENCH_DIR = "/home/EarthExtreme-Bench/data/zip-test"
    local_directory = Path(EE_BENCH_DIR)
    dataset_repo = "zhaoshan/ee-bench_v1.0"

    local_directory.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    dataset_files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")

    for file in dataset_files:
        local_file_path = local_directory / file

        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {file}...")
        hf_hub_download(
            repo_id=dataset_repo,
            filename=file,
            cache_dir=local_directory,
            local_dir=local_directory,
            repo_type="dataset",
        )

    # Decompress each file sequentially

    zip_files = [file for file in dataset_files if file.endswith(".zip")]

    for i, zip_file in enumerate(zip_files):
        print(f"Decompressing {i+1}/{len(zip_files)}: {zip_file}  ...")
        decompress_zip_with_progress(local_directory / zip_file)

    print("Download and decompression process completed.")


if __name__ == "__main__":
    download_benchmark()