import argparse
import os

import torch

from src.trainer import EETask

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster", type=str, default="fire")
    parser.add_argument("--seed", type=int, default=2546)
    parser.add_argument(
        "--mode",
        type=str,
        default="frozen_body",
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
    p.cpu_affinity([32, 33, 34, 35])
    torch.cuda.empty_cache()

    ee_task = EETask(disaster=args.disaster)
    ee_task.train_and_evaluate(
        seed=args.seed,
        mode=args.mode,
        stage=args.stage,
        model_path=None,
        # model_path=f"/home/EarthExtreme-Bench/results/{args.mode}/mit-b0/{args.disaster}/best_model_80000_2024-10-28-01-54",
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
