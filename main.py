import argparse
import os

import torch

from src.trainer import EETask

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster", type=str, default="storm")
    parser.add_argument("--seed", type=int, default=2546)
    parser.add_argument(
        "--mode",
        type=str,
        default="fully_finetune",
        choices=["fully_finetune", "random"],
    )
    parser.add_argument(
        "--stage", type=str, default="test", choices=["train_test", "test"]
    )
    args = parser.parse_args()

    import psutil

    # Get the current process
    p = psutil.Process(os.getpid())
    # Set the CPU affinity (limit to specific CPUs, e.g., CPUs 0 and 1)
    p.cpu_affinity([32, 33, 34, 35])

    ee_task = EETask(disaster=args.disaster)
    ee_task.train_and_evaluate(
        seed=args.seed,
        mode=args.mode,
        stage=args.stage,
        model_path=f"/home/EarthExtreme-Bench/results/{args.mode}/aurora/{args.disaster}/best_model_80000_2024-10-28-01-54",
    )
    # mit-b0/best_model_78_2024-09-06-16-54
    # dataloader = ee_task.get_loader()
    # train_loader = dataloader.test_dataloader()
    # print(len(train_loader))
    # # x = next(iter(test_loader))
    # for id, train_data in enumerate(train_loader):
    #     for key, val in train_data.items():
    #         print(key, val.shape if isinstance(val, torch.Tensor) else val)
    #     break
