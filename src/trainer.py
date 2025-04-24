import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

from .utils.logging_utils import get_logger

sys.path.insert(0, "/home/EarthExtreme-Bench")
from config.settings import settings
from dotenv import load_dotenv
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


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
    def __init__(self, disaster, model_name):
        self.disaster = disaster
        self.model_name = model_name
        self.disaster_data = self.get_loader()
        self.trainer = self.get_trainer()


    def get_loader(self):
        from .dataset.image_dataloader import IMGDataloader
        from .dataset.sequence_dataloader import SEQDataloader

        loader_mappings = {
            "heatwave": IMGDataloader,
            "coldwave": IMGDataloader,
            "tropicalCyclone": IMGDataloader,
            "fire": IMGDataloader,
            "flood": IMGDataloader,
            "storm": SEQDataloader,
            "expcp": SEQDataloader,
        }
        disaster_data = loader_mappings[self.disaster](disaster=self.disaster)
        return disaster_data

    def get_trainer(self):
        if "aurora" in self.model_name:
            from .pcp_train_and_test import ExpcpTrain, StormTrain
            from .tc_train_and_test import TropicalCycloneTrain
            from .t2m_train_and_test import (
                ExtremeTemperatureTrain,
            )
        else:
            from .img_train_and_test import ExtremeTemperatureTrain
            from .seq_train_and_test import ExpcpTrain, StormTrain
        from .img_train_and_test import (
                FireTrain,
                FloodTrain,
            )
        trainers = {
            "heatwave": ExtremeTemperatureTrain,
            "coldwave": ExtremeTemperatureTrain,
            "fire": FireTrain,
            "flood": FloodTrain,
            # "storm": StormTrain,
            # "expcp": ExpcpTrain,
            # "tropicalCyclone": TropicalCycloneTrain,
        }
        trainer = trainers[self.disaster](disaster=self.disaster)
        return trainer

    def train_and_evaluate(self, seed, mode, stage="train_test", model_path=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CURR_FOLDER_PATH = Path(__file__).parent.parent  # "/home/EarthExtreme-Bench"
        torch.set_num_threads(8)
        set_seed(seed)
        config = settings[self.disaster]
        # checkpoint = torch.load('/home/data_storage_home/data/disaster/pretrained_model/Prithvi_100M.pt')
        SAVE_PATH = (
            CURR_FOLDER_PATH
            / "results"
            / mode
            / self.model_name.split("/")[-1]
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
        logger.info(f"Seed: {seed}")

        # model
        input_dim = config["model"].get("input_dim")
        output_dim = config["model"].get("output_dim")
        # model_name = config["model"]["name"]
        model_name = self.model_name

        # Please register new model here
        model_import_paths = {
            # CV models
            "openmmlab/upernet-convnext-tiny": "Baseline",
            "nvidia/mit-b0": "Baseline",
            "unet": "Baseline",
            # Weather models
            "microsoft/aurora": "Weather",
            "microsoft/aurora_pcp": "Weather",
            "microsoft/aurora_t2m": "Weather",
            "microsoft/climax": "Weather",
            # RS models
            "xshadow/dofa": "RS",
            "xshadow/dofa_upernet": "RS",
            "ibm-nasa-geospatial/prithvi": "RS",
            "ibm-nasa-geospatial/prithvi-2_upernet": "RS",
            "stanford/satmae": "RS",
        }

        if model_name in model_import_paths:
            module_path = f"src.models.model_{model_import_paths[model_name]}"
            from importlib import import_module

            BaselineNet = getattr(import_module(module_path), "BaselineNet")
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        model = BaselineNet(
            disaster=self.disaster,
            model_name=model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            img_size=config["dataloader"].get("img_size"),
            num_frames=config["dataloader"].get("in_seq_length"),
            wave_list=config.wave_list,
            freezing_body=True if mode == "frozen_body" else False,
            logger=logger,
        )

        if mode == "random":
            model._initialize_weights()
            for _, param in model.named_parameters():
                param.requires_grad = True
            logger.info("random initialized model....")

        logger.info(model)
        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Log the total number of parameters
        logger.info(f"Total number of parameters in the model: {total_params}")

        model = model.to(device)

        num_epochs = config["train"]["num_epochs"]
        # infinite sampler - use iteration instead of epoch
        # num_epochs = config["train"][
        #     "num_iterations"
        # ]  # infinite sampler - use iteration instead of epoch

        # optimizer
        # optimizer = torch.optim.SGD(
        #     model.parameters(),
        #     lr=config["train"]["lr"],
        #     weight_decay=config["train"]["weight_decay"],
        # )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, "min", patience=2
        # )
        warm_up_iter = 3
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=config["train"]["lr"] / 100,
            end_factor=1.0,
            total_iters=warm_up_iter,
        )
        # Main learning rate scheduler
        # main_scheduler = CosineAnnealingLR(
        #     optimizer, T_max=num_epochs, eta_min=config["train"]["lr"] / 100
        # )
        main_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=num_epochs)
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warm_up_iter],
        )
        # lr_scheduler = main_scheduler

        loss = config["train"]["loss"]
        patience = config["train"]["patience"]
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 50], gamma=0.5)

        # training
        load_dotenv(dotenv_path="config/.env")
        # reused_dic = torch.load(
        #     "results/frozen_body/dofa/fire/last_model.pth"
        # )
        #
        # model.load_state_dict(reused_dic,strict=True)
        # logger.info("resume training from epoch 8...")

        if stage == "train_test":
            wandb.init(
                project=f"ee_bench-{self.disaster}",
                name=f"{mode}-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                entity=os.getenv("WANDB_ENTITY"),
                dir=os.getenv("WANDB_DIR"),
                tags=os.getenv("WANDB_TAG"),
            )
            best_model_state_dict, best_epoch, last_state, last_epoch = (
                self.trainer.train(
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
            wandb.finish()
        elif stage == "test":
            backup_model_path = Path(model_path)
            model_id = str(backup_model_path).split("/")[-1]
        best_model_state_dict = torch.load(backup_model_path / f"{model_id}.pth")
        logger.info(f"Begin testing {backup_model_path}...")

        msg = model.load_state_dict(best_model_state_dict)
        logger.info(msg)

        # Testing
        scores = self.trainer.test(
            model,
            self.disaster_data,
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


if __name__ == "__main__":
    pass
