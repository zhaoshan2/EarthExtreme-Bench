import argparse
import os
import warnings
import torch

from src.trainer import EETask

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster", type=str, default="flood")
    parser.add_argument("--model_name", type=str, default="stanford/satmae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["fully_finetune", "frozen_body", "random"],
    )
    parser.add_argument(
        "--stage", type=str, default="train_test", choices=["train_test", "test"]
    )
    args = parser.parse_args()

    import psutil

    # Get the current process
    p = psutil.Process(os.getpid())
    # Set the CPU affinity (limit to specific CPUs, e.g., CPUs 0 and 1)
    # p.cpu_affinity([15, 16, 17, 18])
    torch.cuda.empty_cache()

    warnings.filterwarnings(
        "ignore",
        message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.*",
    )
    import tomli
    with open("config/dataset.toml", "rb") as f:
        tomli.load(f)
    ee_task = EETask(disaster=args.disaster, model_name=args.model_name)
    ee_task.train_and_evaluate(
        seed=args.seed,
        mode=args.mode,
        stage=args.stage,
        model_path=f"/home/EarthExtreme-Bench/results/{args.mode}/{args.model_name}/{args.disaster}/best_model_29_2025-04-02-12-50",##
    )
    # unet best_model_49_2024-11-15-09-11
    # upernet-convnext-tiny best_model_49_2024-11-13-15-46
    #
    # dataloader = ee_task.get_loader()
    # train_loader = dataloader.test_dataloader()
    # print(len(train_loader))
    # # x = next(iter(test_loader))
    # for id, train_data in enumerate(train_loader):
    #     for key, val in train_data.items():
    #         print(key, val.shape if isinstance(val, torch.Tensor) else val)
    #     break
