import sys

sys.path.insert(0, "/home/EarthExtreme-Bench")
import torch
import os
import random
from pathlib import Path
from datetime import datetime
from utils.logging_utils import get_logger
import wandb
import numpy as np

from config.settings import settings
from dotenv import load_dotenv


def set_seed(seed):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable the inbuilt cudnn auto-tune


class EETask:
    def __init__(self, disaster):
        self.disaster = disaster
        self.disaster_data = self.get_loader()
        self.trainer = self.get_trainer()

    def get_loader(self):
        from utils.dataset.image_dataloader import IMGDataloader
        from utils.dataset.sequence_dataloader import SEQDataloader

        loader_mappings = {
            "heatwave": IMGDataloader,
            "coldwave": IMGDataloader,
            "tropicalCylone": IMGDataloader,
            "fire": IMGDataloader,
            "flood": IMGDataloader,
            "storm": SEQDataloader,
            "expcp": SEQDataloader,
        }
        disaster_data = loader_mappings[self.disaster](disaster=self.disaster)
        return disaster_data

    def get_trainer(self):
        from utils.trainer.img_train_and_test import (
            ExtremeTemperatureTrain,
            FireTrain,
            FloodTrain,
        )
        from utils.trainer.seq_train_and_test import ExpcpTrain, StormTrain

        trainers = {
            "heatwave": ExtremeTemperatureTrain,
            "coldwave": ExtremeTemperatureTrain,
            "fire": FireTrain,
            "flood": FloodTrain,
            "storm": StormTrain,
            "expcp": ExpcpTrain,
        }
        trainer = trainers[self.disaster](disaster=self.disaster)
        return trainer

    def train_and_evaluate(self, seed, mode):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CURR_FOLDER_PATH = Path(__file__).parent.parent  # "/home/EarthExtreme-Bench"
        torch.set_num_threads(4)
        set_seed(seed)
        config = settings[self.disaster]
        # checkpoint = torch.load('/home/data_storage_home/data/disaster/pretrained_model/Prithvi_100M.pt')
        SAVE_PATH = (
            CURR_FOLDER_PATH
            / "results"
            / mode
            / config["model"]["name"].split("/")[-1]
            / self.disaster
        )
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        # Create the logger
        log_path = SAVE_PATH / "training.log"
        if os.path.exists(log_path):
            os.remove(log_path)
        logger = get_logger(log_dir=SAVE_PATH)
        logger.info(f"{config}")

        # model
        input_dim = config["model"].get("input_dim")
        output_dim = config["model"].get("output_dim")
        model_name = config["model"]["name"]

        if mode == "random":
            from models.model_Baseline_vision_random import BaselineNet
        elif mode == "fully_supervised":
            from models.model_Baseline_vision import BaselineNet

        model = BaselineNet(
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=model_name,
        )
        model = model.to(device)

        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=2
        )
        num_epochs = (
            config["train"].get("num_epochs")
            if not None
            else config["train"]["num_iterations"]
        )  # infinite sampler - use iteration instead of epoch
        loss = config["train"]["loss"]
        patience = config["train"]["patience"]
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 50], gamma=0.5)

        # training
        load_dotenv(dotenv_path="config/.env")
        wandb.init(
            project=f"ee_bench-{self.disaster}",
            name=f"{mode}-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            entity=os.getenv("WANDB_ENTITY"),
            dir=os.getenv("WANDB_DIR"),
            tags=os.getenv("WANDB_TAG"),
        )

        best_model_state_dict, best_epoch, last_state, last_epoch = self.trainer.train(
            model,
            data=self.disaster_data,
            device=device,
            ckp_path=SAVE_PATH,
            num_epochs=num_epochs,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            patience=patience,
            logger=logger,
        )

        logger.info(f"The best model is saved at epoch {best_epoch}")
        # Rename the best_model_state_dict with its epoch and the current time
        now = datetime.now()
        # Format the date and time as a string: "year-month-day-hour-minutes"
        formatted_datetime = now.strftime("%Y-%m-%d-%H-%M")
        model_id = f"best_model_{best_epoch}_{formatted_datetime}"  # current

        backup_model_path = SAVE_PATH / model_id

        if not os.path.exists(backup_model_path):
            os.makedirs(backup_model_path)
        # Reload the best stat dic
        with open(backup_model_path / f"{model_id}.pth", "wb") as b_f:
            torch.save(best_model_state_dict, b_f)
        with open(backup_model_path / f"last_epoch{last_epoch}.pth", "wb") as l_f:
            torch.save(last_state, l_f)
        # backup_model_path = Path(
        #     "/home/EarthExtreme-Bench/results/upernet-convnext-tiny/storm/best_model_4000_2024-09-25-03-58"
        # )
        # model_id = str(backup_model_path).split("/")[-1]
        best_model_state_dict = torch.load(backup_model_path / f"{model_id}.pth")
        logger.info(f"Begin testing {backup_model_path}...")
        msg = model.load_state_dict(best_model_state_dict)
        logger.info(msg)

        # Testing
        test_loader = self.disaster_data.test_dataloader()
        logger.info(f"length of test loader {len(test_loader)}")
        scores = self.trainer.test(
            model,
            test_loader,
            device,
            stats=config["normalization"],
            save_path=SAVE_PATH,
            model_id=model_id,
            seq_len=config["dataloader"].get("out_seq_length"),
        )
        logger.info("Finish testing!")
        logger.info(scores)

        # Open the .log file in read mode and the .txt file in write mode
        dst_log = SAVE_PATH / model_id / "training.log"
        src_log = SAVE_PATH / "training.log"
        with open(src_log, "r") as log_file, open(dst_log, "a") as txt_file:
            # Read from the .log file and write to the .txt file
            contents = log_file.read()
            txt_file.write(contents)
        wandb.finish()


if __name__ == "__main__":
    pass
