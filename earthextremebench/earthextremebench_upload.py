from huggingface_hub import HfApi
import os
from pathlib import Path
def upload_wewather_dataset():
    LOCAL_DATA_DIR = Path(__file__).parent.parent / 'data' / 'weather'
    api = HfApi()
    #api.create_repo(repo_id="ee-bench_v1.0", repo_type="dataset", private=True)
    for root, subdirs, _ in os.walk(LOCAL_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith(".zip"):
                    print(f"Uploading {os.path.join(root, subdir, filename)}...")

                    api.upload_file(
                        path_or_fileobj=os.path.join(root, subdir, filename),
                        path_in_repo=f"data/weather/{filename}",
                        repo_id="zhaoshan/ee-bench_v1.0",
                        repo_type="dataset"
                    )
def upload_eo_dataset():
    LOCAL_DATA_DIR = Path(__file__).parent.parent.parent / 'data_storage_home/data/disaster' / 'data'

    if not os.path.exists(LOCAL_DATA_DIR):
        print("The local path does not exist.")
    api = HfApi()
    #api.create_repo(repo_id="ee-bench_v1.0", repo_type="dataset", private=True)
    for root, subdirs, _ in os.walk(LOCAL_DATA_DIR):
        for subdir in subdirs:
            for file in os.listdir(os.path.join(root, subdir)):
                filename = os.fsdecode(file)
                if filename.endswith(".zip"):
                    print(f"Uploading {os.path.join(root, subdir, filename)}...")

                    api.upload_file(
                        path_or_fileobj=os.path.join(root, subdir, filename),
                        path_in_repo=f"data/eo/{filename}",
                        repo_id="zhaoshan/ee-bench_v1.0",
                        repo_type="dataset"
                    )
if __name__ == "__main__":
    upload_wewather_dataset()