import sys

import torch

sys.path.insert(0, "/home/EarthExtreme-Bench")
import argparse
import os
import random
from pathlib import Path

import numpy as np
from model_Baseline_vision import BaselineNet


def set_seed(seed):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable the inbuilt cudnn auto-tune


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster", type=str, default="fire")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CURR_FOLDER_PATH = Path(__file__).parent.parent  # "/home/EarthExtreme-Bench"
    torch.set_num_threads(2)
    set_seed(42)
    import psutil

    # Get the current process
    p = psutil.Process(os.getpid())

    # Set the CPU affinity (limit to specific CPUs, e.g., CPUs 0 and 1)
    p.cpu_affinity([34, 35])

    # checkpoint = torch.load('/home/data_storage_home/data/disaster/pretrained_model/Prithvi_100M.pt')

    # model_name = 'openmmlab/upernet-convnext-tiny'
    model_name = "nvidia/mit-b0"
    SAVE_PATH = CURR_FOLDER_PATH / "results" / "mit-b0"
    # model.load_state_dict(torch.load(SAVE_PATH / 'heatwave' / 'best_model_200.pth'))

    if args.disaster == "heatwave":

        from utils.dataset.era5_dataloader import ERA5Dataloader
        from utils.trainer.era5_train_and_test import test, train

        # dataset
        heatwave = ERA5Dataloader(
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            horizon=28,
            chip_size=512,
            val_ratio=0.4,
            data_path="/home/EarthExtreme-Bench/data/weather",
            persistent_workers=False,
        )

        train_loader, records = heatwave.train_dataloader()
        print("length of training loader", len(train_loader))
        val_loader, _ = heatwave.val_dataloader()
        print("length of validation loader", len(val_loader))
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        # model
        model = BaselineNet(input_dim=4, output_dim=1, model_name=model_name)
        model = model.to(device)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 50], gamma=0.5)

        # training
        best_model_state_dict, best_epoch = train(
            model,
            train_loader,
            val_loader,
            device,
            save_path=SAVE_PATH,
            num_epochs=100,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        # print(f"the model is saved at epoch {best_epoch}")
        # rename the best_model_state_dict with its epoch
        model_id = f"best_model"  # current
        # ckp_path = SAVE_PATH / args.disaster / f"{model_id}.pth"
        #
        # best_model_state_dict = torch.load(SAVE_PATH / args.disaster / f'{model_id}.pth')
        msg = model.load_state_dict(best_model_state_dict)
        print(msg)
        # testing
        test_loader, records = heatwave.test_dataloader()
        print("length of test loader", len(test_loader))
        _ = test(
            model,
            test_loader,
            device,
            stats=records.mean_std_dic,
            save_path=SAVE_PATH,
            model_id=model_id,
        )

    elif args.disaster == "fire":
        from utils.dataset.multispectral_dataloader import \
            MultiSpectralDataloader
        from utils.trainer.multispectral_train_and_test import test, train

        # dataset
        burned = MultiSpectralDataloader(
            batch_size=8,
            num_workers=0,
            pin_memory=True,
            chip_size=512,
            data_path="/home/EarthExtreme-Bench/data/eo/hls_burn_scars",
            val_ratio=0.2,
            persistent_workers=False,
            transform=None,
            disaster=args.disaster,
        )
        train_loader = burned.train_dataloader()
        print(f"Training set has the length of {len(train_loader)}")
        val_loader = burned.val_dataloader()
        print(f"Validation set has the length of {len(val_loader)}")
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        # define model
        model = BaselineNet(input_dim=6, output_dim=2, model_name=model_name)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        best_model_state_dict, best_epoch = train(
            model,
            train_loader,
            val_loader,
            device,
            save_path=SAVE_PATH,
            num_epochs=100,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            disaster=args.disaster,
        )
        # best_epoch = 20
        model_id = f"best_model_{best_epoch}"
        # best_model_state_dict = torch.load(SAVE_PATH / args.disaster / f"{model_id}.pth")
        msg = model.load_state_dict(best_model_state_dict)
        print(msg)

        test_loader = burned.test_dataloader()
        _ = test(
            model,
            test_loader,
            device,
            save_path=SAVE_PATH,
            disaster=args.disaster,
            model_id=model_id,
        )

    elif args.disaster == "flood":
        from utils.dataset.multispectral_dataloader import \
            MultiSpectralDataloader
        from utils.trainer.multispectral_train_and_test import test, train

        # dataset
        flood = MultiSpectralDataloader(
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            chip_size=512,
            data_path="/home/EarthExtreme-Bench/data/eo/flood",
            val_ratio=0.2,
            persistent_workers=False,
            transform=None,
            disaster=args.disaster,
        )
        # train_loader = flood.train_dataloader()
        # print(f"Training set has the length of {len(train_loader)}")
        # val_loader = flood.val_dataloader()
        # print(f"Validation set has the length of {len(val_loader)}")
        # if not os.path.exists(SAVE_PATH):
        #     os.mkdir(SAVE_PATH)

        # define model
        model = BaselineNet(input_dim=8, output_dim=3, model_name=model_name)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        # best_model_state_dict, best_epoch = train(model, train_loader, val_loader, device, save_path=SAVE_PATH,
        #                                num_epochs=100, optimizer=optimizer,
        #                                    lr_scheduler=lr_scheduler, disaster=args.disaster)
        best_epoch = 85
        model_id = f"best_model_{best_epoch}"
        best_model_state_dict = torch.load(
            SAVE_PATH / args.disaster / f"{model_id}.pth"
        )
        msg = model.load_state_dict(best_model_state_dict)
        print(msg)

        test_loader = flood.test_dataloader()
        _ = test(
            model,
            test_loader,
            device,
            save_path=SAVE_PATH,
            disaster=args.disaster,
            model_id=model_id,
        )
