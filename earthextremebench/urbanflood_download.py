from datasets import load_dataset
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
def download_benchmark():
    UR_BENCH_DIR = "/home/EarthExtreme-Bench/data/eo/flood/"
    local_directory = Path(UR_BENCH_DIR)
    dataset_repo = "S1Floodbenchmark/UrbanSARFloods_v1"

    local_directory.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    dataset_files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")

    for file in dataset_files:
        if file.startswith('03_FU/SAR'):
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
if __name__ == "__main__":
    download_benchmark()