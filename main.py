import wandb
from dotenv import load_dotenv
from models.ee_models import EETask
import argparse
import os
import torch

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster", type=str, default="expcp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="fully_finetune",
        choices=["fully_finetune", "random"],
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

    ee_task = EETask(disaster=args.disaster)
    ee_task.train_and_evaluate(
        seed=args.seed,
        mode=args.mode,
        stage=args.stage,
        model_path=f"/home/EarthExtreme-Bench/results/{args.mode}/aurora/{args.disaster}/best_model_5000_2024-10-16-04-50",
    )

    # dataloader = ee_task.get_loader()
    # train_loader = dataloader.test_dataloader()
    # print(len(train_loader))
    # # x = next(iter(test_loader))
    # for id, train_data in enumerate(train_loader):
    #     for key, val in train_data.items():
    #         print(key, val.shape if isinstance(val, torch.Tensor) else val)
    #     break
