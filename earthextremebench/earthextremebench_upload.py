import os
from pathlib import Path

from huggingface_hub import HfApi


def upload_weather_dataset():
    LOCAL_DATA_DIR = Path(__file__).parent.parent / "data" / "weather"
    api = HfApi()
    # api.create_repo(repo_id="ee-bench_v1.0", repo_type="dataset", private=True)
    new_branch = "stable"

    # Create the branch
    api.create_branch(
        repo_id="zhaoshan/ee-bench_v1.0", repo_type="dataset", branch=new_branch
    )
    for root, subdirs, files in os.walk(LOCAL_DATA_DIR):
        for file in files:
            filename = os.fsdecode(file)
            if filename.endswith(".zip"):
                print(f"Uploading {os.path.join(root, filename)}...")

                api.upload_file(
                    path_or_fileobj=os.path.join(root, filename),
                    path_in_repo=f"data/weather/{filename}",
                    repo_id="zhaoshan/ee-bench_v1.0",
                    repo_type="dataset",
                )

    file_paths = [
        "data/weather/coldwave-daily.zip",
        "data/weather/heatwave-daily.zip",
        "data/weather/storm-minutes.zip",
        "data/weather/tropicalCyclone-hourly.zip",
    ]

    # Delete each file
    for file_path in file_paths:
        api.delete_file(
            repo_id="zhaoshan/ee-bench_v1.0",
            path_in_repo=file_path,
            repo_type="dataset",
            revision="stable",
        )


def upload_eo_dataset():
    LOCAL_DATA_DIR = Path(__file__).parent.parent / "data" / "eo"

    if not os.path.exists(LOCAL_DATA_DIR):
        print("The local path does not exist.")
    api = HfApi()
    # api.create_repo(repo_id="ee-bench_v1.0", repo_type="dataset", private=True)
    # for root, subdirs, files in os.walk(LOCAL_DATA_DIR):
    #     for file in files:
    #         filename = os.fsdecode(file)
    #         if filename.endswith(".zip"):
    #             print(f"Uploading {os.path.join(root, filename)}...")
    #
    #             api.upload_file(
    #                 path_or_fileobj=os.path.join(root, filename),
    #                 path_in_repo=f"data/eo/{filename}",
    #                 repo_id="zhaoshan/ee-bench_v1.0",
    #                 repo_type="dataset",
    #                 revision="stable",
    #             )

    file_paths = ["data/eo/hls_burn_scars.zip"]

    # Delete each file
    for file_path in file_paths:
        api.delete_file(
            repo_id="zhaoshan/ee-bench_v1.0",
            path_in_repo=file_path,
            repo_type="dataset",
            revision="stable",
        )


def upload_masks():
    LOCAL_DATA_DIR = Path(__file__).parent.parent / "data" / "masks"

    if not os.path.exists(LOCAL_DATA_DIR):
        print("The local path does not exist.")
    api = HfApi()
    # api.create_repo(repo_id="ee-bench_v1.0", repo_type="dataset", private=True)
    for root, subdirs, files in os.walk(LOCAL_DATA_DIR):
        for file in files:
            filename = os.fsdecode(file)
            # if filename.endswith(".zip"):
            print(f"Uploading {os.path.join(root, filename)}...")

            api.upload_file(
                path_or_fileobj=os.path.join(root, filename),
                path_in_repo=f"data/masks/{filename}",
                repo_id="zhaoshan/ee-bench_v1.0",
                repo_type="dataset",
                revision="stable",
            )


if __name__ == "__main__":
    upload_eo_dataset()
