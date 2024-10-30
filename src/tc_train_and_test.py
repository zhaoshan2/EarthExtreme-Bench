import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler

from .models.model_components.aurora.aurora import Batch, Metadata
from .utils import logging_utils, score_utils

sys.path.insert(0, "/home/EarthExtreme-Bench")
from config.settings import settings


class TropicalCycloneTrain:
    def __init__(self, disaster):
        self.disaster = "tropicalCyclone"
        self.loss_mapping = {
            "l1": nn.L1Loss(reduction="mean"),
        }
        self.sur_weights = torch.Tensor(
            settings[self.disaster].train.surface_weights.to_list()
        ).view(1, 3, 1, 1)
        self.atm_weights = torch.Tensor(
            settings[self.disaster].train.upper_weights.to_list()
        ).view(1, 3, 5, 1, 1)

    def prepare_batch(self, input, global_metas):

        local_metas = input["meta_info"]
        batch_size = input["x"].shape[0]

        batch = Batch(
            # The aurora requires variable name differs from the era5 name
            surf_vars={
                k: input["x"][:, i, ...]
                for i, k in enumerate(
                    ["msl", "10u", "10v"]
                )  # global_metas["sur_variables"]
            },
            static_vars={
                k: input["mask"][0, i, ...]
                # k: F.pad(input["mask"][0, i, ...], (0, 0, 1, 0))
                for i, k in enumerate(global_metas["mask_variables"])
            },
            atmos_vars={
                k: input["x_upper"][:, i, ...]
                for i, k in enumerate(global_metas["atm_variables"])
            },
            metadata=Metadata(
                lat=torch.Tensor(
                    [
                        round(x * 4) / 4
                        for x in np.linspace(
                            local_metas["latitude"].item(),
                            local_metas["latitude"].item()
                            - local_metas["resolution"].item() * (input["x"].shape[-2]),
                            input["x"].shape[-2],
                        )
                    ]
                ),
                lon=torch.Tensor(
                    [
                        round(x * 4) / 4
                        for x in np.linspace(
                            local_metas["longitude"].item(),
                            local_metas["longitude"].item()
                            + local_metas["resolution"].item() * input["x"].shape[-1],
                            input["x"].shape[-1],
                        )
                    ]
                ),
                time=tuple(
                    datetime.strptime(local_metas["input_time"][i], "%Y-%m-%d %H:%M")
                    for i in range(batch_size)
                ),
                atmos_levels=global_metas["pressures"],
                # To do, infer the rollout steps from the input and target time
                # rollout_step = local_metas["target_time"] - local_metas["input_time"]
            ),
        )

        return batch

    def train(
        self,
        model,
        data,
        device,
        ckp_path: Path,
        num_epochs,
        optimizer,
        lr_scheduler,
        loss,
        patience=20,
        logger=None,
        **args,
    ):
        """Training code"""
        # Prepare for the optimizer and scheduler
        # scaler = GradScaler()
        train_loader, META_INFO = data.train_dataloader()
        val_loader, _ = data.val_dataloader()
        val_interval = 1000
        # Loss function
        criterion = self.loss_mapping[loss]

        best_loss = np.inf
        total_iter = 0
        stop_training = False
        while True:
            if stop_training or total_iter >= num_epochs:
                break
            try:
                epoch_loss, epoch_loss_sur, epoch_loss_atm = 0.0, 0.0, 0.0

                for id, train_data in enumerate(train_loader):
                    total_iter += 1
                    # Call the model and get the output
                    # train_sur: (b,n,t,h,w), train_atmos: (b,n,t,z,h,w)
                    batch = self.prepare_batch(train_data, META_INFO)
                    batch = batch.to(device)

                    optimizer.zero_grad()

                    model.train()
                    logits = model(
                        batch
                    )  # (upsampled) logits with the same w,h as inputs (b,c_out,h,w)

                    # put batch into the stacked tensor
                    logits_sur = [
                        logits.surf_vars[var]
                        .clone()
                        .requires_grad_(True)
                        .unsqueeze(
                            1
                        )  # logits.surf_vars[var] (1, 1, h, w) -> (b,1,t,h,w)
                        for var in batch.surf_vars.keys()
                    ]
                    # squeeze the time dimension
                    logits_sur = torch.cat(logits_sur, dim=1).squeeze(
                        2
                    )  # [1, 3, 1, 96, 96] -> [1, 3, 96, 96]

                    logits_atm = [
                        logits.atmos_vars[var]
                        .clone()
                        .requires_grad_(True)
                        .unsqueeze(
                            1
                        )  # logits.atmos_vars[var] (1, 1,z,h, w) -> (b,1,1,z,h,w)
                        for var in batch.atmos_vars.keys()
                    ]
                    # squeeze the time dimension
                    logits_atm = torch.cat(logits_atm, dim=1).squeeze(
                        2
                    )  # (b,3,1,z,h,w) -> (b,3,1,z,h,w)

                    y_train = train_data["y"].to(device).squeeze(2)
                    y_atm_train = train_data["y_upper"].to(device).squeeze(2)

                    loss_sur = criterion(
                        logits_sur * self.sur_weights.to(device),
                        y_train * self.sur_weights.to(device),
                    )
                    loss_atm = criterion(
                        logits_atm * self.atm_weights.to(device),
                        y_atm_train * self.atm_weights.to(device),
                    )

                    loss = 0.25 * loss_sur + loss_atm

                    # Call the backward algorithm and calculate the gratitude of parameters
                    # scaler.scale(loss).backward()
                    loss.backward()

                    # Update model parameters with AdamW optimizer
                    # scaler.step(optimizer)
                    # scaler.update()
                    optimizer.step()
                    lr_scheduler.step()

                    epoch_loss_sur += loss_sur.item()
                    epoch_loss_atm += loss_atm.item()
                    epoch_loss = epoch_loss_sur + epoch_loss_atm

                    wandb.log(
                        {
                            "train sur loss": loss_sur.item(),
                            "train atm loss": loss_atm.item(),
                            "loss": loss_sur.item() + loss_atm.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                    # Save the model every 100 epochs
                    if total_iter % 100 == 0:
                        last_state = {
                            key: value.cpu()
                            for key, value in model.state_dict().items()
                        }
                        file_path = os.path.join(ckp_path, "last_model.pth")
                        with open(file_path, "wb") as f:
                            torch.save(last_state, f)

                    # epoch_loss /= len(train_loader)
                    # epoch_loss_sur /= len(train_loader)
                    # epoch_loss_atm /= len(train_loader)

                    # Validate
                    if total_iter % val_interval == 0:
                        logger.info(
                            "iter {} : {:.3f}".format(
                                total_iter, epoch_loss / val_interval
                            )
                        )
                        epoch_loss, epoch_loss_sur, epoch_loss_atm = 0.0, 0.0, 0.0
                        with torch.no_grad():
                            loss_val = 0
                            for id, val_data in enumerate(val_loader):
                                batch = self.prepare_batch(val_data, META_INFO)
                                batch = batch.to(device)

                                logits = model(batch)
                                logits_sur = [
                                    logits.surf_vars[var].clone().unsqueeze(1)
                                    for var in batch.surf_vars.keys()
                                ]
                                logits_sur = torch.cat(logits_sur, dim=1).squeeze(
                                    2
                                )  # [1, 3, 1, w, h] -> [1, 3, w, h]

                                logits_atm = [
                                    logits.atmos_vars[var].clone().unsqueeze(1)
                                    for var in batch.atmos_vars.keys()
                                ]
                                logits_atm = torch.cat(logits_atm, dim=1).squeeze(
                                    2
                                )  # [1, 3, 1, 5, w, h] -> [1, 3, 5, w, h]

                                y_val = val_data["y"].to(device).squeeze(2)
                                y_atm_val = val_data["y_upper"].to(device).squeeze(2)
                                loss_sur = criterion(
                                    logits_sur * self.sur_weights.to(device),
                                    y_val * self.sur_weights.to(device),
                                )
                                loss_atm = criterion(
                                    logits_atm * self.atm_weights.to(device),
                                    y_atm_val * self.atm_weights.to(device),
                                )

                                loss = 0.25 * loss_sur + loss_atm

                                loss_val += loss.item()

                            loss_val /= len(val_loader)
                            wandb.log({"val loss": loss_val})
                            logger.info(
                                "Val loss {} : {:.3f}".format(total_iter, loss_val)
                            )
                            if loss_val < best_loss:
                                best_loss = loss_val
                                best_iter = total_iter
                                best_state = {
                                    key: value.cpu()
                                    for key, value in model.state_dict().items()
                                }
                                file_path = os.path.join(ckp_path, "best_model.pth")
                                with open(file_path, "wb") as f:
                                    torch.save(best_state, f)
                                    logger.info(
                                        f"Saving the best model at iteration {best_iter} to {file_path}...."
                                    )
                            elif total_iter >= best_iter + patience * val_interval:
                                print("Early stopping triggered.")
                                stop_training = True
                                break
            except KeyboardInterrupt:
                return (
                    best_state,
                    best_iter,
                    last_state,
                    total_iter,
                )
        return best_state, best_iter, last_state, total_iter

    def test(self, model, data, device, stats, save_path, model_id, **kwargs):
        # turn off gradient tracking for evaluation
        test_loader, META_INFO = data.test_dataloader()
        rmse_sur, rmse_atmos, acc_sur, acc_atmos = dict(), dict(), dict(), dict()
        nz = len(META_INFO["atm_variables"]) * len(META_INFO["pressures"])
        criterion = nn.L1Loss()
        total_loss = 0
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                batch = self.prepare_batch(test_data, META_INFO)
                model.eval()
                logits = model(batch)
                logits_sur = [
                    logits.surf_vars[var].clone().detach().unsqueeze(1)
                    for var in batch.surf_vars.keys()
                ]
                logits_sur = torch.cat(logits_sur, dim=1).squeeze(
                    2
                )  # [1, 3, w, h] # ignore the time dimension

                logits_atm = [
                    logits.atmos_vars[var].clone().detach().unsqueeze(1)
                    for var in batch.atmos_vars.keys()
                ]
                logits_atm = torch.cat(logits_atm, dim=1).squeeze(2)  # [1, 3, z, w, h]

                y_test = test_data["y"].to(device).squeeze(2)
                y_atm_test = test_data["y_upper"].to(device).squeeze(2)
                loss_sur = criterion(
                    logits_sur * self.sur_weights.to(device),
                    y_test * self.sur_weights.to(device),
                )
                loss_atm = criterion(
                    logits_atm * self.atm_weights.to(device),
                    y_atm_test * self.atm_weights.to(device),
                )

                loss = 0.25 * loss_sur + loss_atm

                total_loss += loss
                time_str = datetime.strptime(
                    test_data["meta_info"]["target_time"][0], "%Y-%m-%d %H:%M"
                ).strftime("%Y-%m-%d_%H%M")
                target_time = f"{test_data['disno'][0]}-{time_str}"
                surface_means = (
                    torch.Tensor(stats["surface_means"].to_list())
                    .view(1, len(META_INFO["sur_variables"]), 1, 1)
                    .to(device)
                )
                surface_stds = (
                    torch.Tensor(stats["surface_stds"].to_list())
                    .view(1, len(META_INFO["sur_variables"]), 1, 1)
                    .to(device)
                )

                atmos_means = (
                    torch.Tensor(stats["upper_means"].to_list())
                    .view(
                        1,
                        len(META_INFO["atm_variables"]),
                        len(META_INFO["pressures"]),
                        1,
                        1,
                    )
                    .to(device)
                )

                atmos_stds = (
                    torch.Tensor(stats["upper_stds"].to_list())
                    .view(
                        1,
                        len(META_INFO["atm_variables"]),
                        len(META_INFO["pressures"]),
                        1,
                        1,
                    )
                    .to(device)
                )

                acc_sur[target_time] = (
                    score_utils.unweighted_acc_torch(
                        (logits_sur - surface_means) / surface_stds,
                        (y_test - surface_means) / surface_stds,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )  # [1, 3, w, h]

                acc_atmos[target_time] = (
                    score_utils.unweighted_acc_torch(
                        ((logits_atm - atmos_means) / atmos_stds).view(
                            1, nz, logits_atm.shape[-2], logits_atm.shape[-1]
                        ),
                        ((y_atm_test - atmos_means) / atmos_stds).view(
                            1, nz, logits_atm.shape[-2], logits_atm.shape[-1]
                        ),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                # rmse
                csv_path = save_path / model_id / "csv"
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)

                rmse_sur[target_time] = (
                    score_utils.unweighted_rmse_torch(logits_sur, y_test)
                    .detach()
                    .cpu()
                    .numpy()
                )  # returns varaible-wise score mean over b,w,h
                rmse_atmos[target_time] = (
                    score_utils.unweighted_rmse_torch(
                        logits_atm.view(
                            1, nz, logits_atm.shape[-2], logits_atm.shape[-1]
                        ),
                        y_atm_test.view(
                            1, nz, logits_atm.shape[-2], logits_atm.shape[-1]
                        ),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )  # returns varaible-wise score mean over b,w,h
                # visualize the last frame
                # put all tensors to cpu
                if id % 100 == 0:
                    target_test = y_test.detach().cpu().numpy()
                    target_atm_test = y_atm_test.detach().cpu().numpy()
                    # (1, t, 17, 32)
                    var = "msl"
                    x = batch.surf_vars[var].detach().cpu().numpy()
                    output_test = logits.surf_vars["msl"].detach().cpu().numpy()
                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    s, l = np.amin(target_test[0, 0]), np.amax(target_test[0, 0])
                    im = axes[0].imshow(x[0, 0], vmin=s, vmax=l)
                    plt.colorbar(im, ax=axes[0])
                    axes[0].set_title(f"input_{var}")

                    im = axes[1].imshow(target_test[0, 0])
                    plt.colorbar(im, ax=axes[1])
                    axes[1].set_title(f"target_{var}")

                    im = axes[2].imshow(output_test[0, 0], vmin=s, vmax=l)
                    plt.colorbar(im, ax=axes[2])
                    axes[2].set_title(f"pred_{var}")

                    png_path = save_path / model_id / "png"
                    if not os.path.exists(png_path):
                        os.makedirs(png_path)
                    plt.savefig(f"{png_path}/test_pred_{target_time}_{var}.png")

                    var = "v"  # (1, t, z, 17, 32)
                    pz = 500  # 3rd pressure
                    x = batch.atmos_vars[var].detach().cpu().numpy()
                    output_test = logits.atmos_vars["v"].detach().cpu().numpy()

                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    s, l = np.amin(target_atm_test[0, 2, 3]), np.amax(
                        target_atm_test[0, 2, 3]
                    )
                    im = axes[0].imshow(x[0, 0, 3, ...], vmin=s, vmax=l)
                    plt.colorbar(im, ax=axes[0])
                    axes[0].set_title(f"input_{var}")

                    im = axes[1].imshow(target_atm_test[0, 2, 3, ...])
                    plt.colorbar(im, ax=axes[1])
                    axes[1].set_title(f"target_{var}")

                    im = axes[2].imshow(output_test[0, 0, 3], vmin=s, vmax=l)
                    plt.colorbar(im, ax=axes[2])
                    axes[2].set_title(f"pred_{var}")

                    png_path = save_path / model_id / "png"
                    if not os.path.exists(png_path):
                        os.makedirs(png_path)
                    plt.savefig(f"{png_path}/test_pred_{target_time}_{var}_{pz}.png")

                    plt.close(fig)
                # Save rmses to csv
                total_loss += loss
        total_loss = total_loss / id
        logging_utils.save_errorScores(csv_path, acc_sur, "acc_sur")
        logging_utils.save_errorScores(csv_path, acc_atmos, "acc_atmos")
        logging_utils.save_errorScores(csv_path, rmse_sur, "rmse_sur")
        logging_utils.save_errorScores(csv_path, rmse_atmos, "rmse_atmos")
        return total_loss


if __name__ == "__main__":
    trainer = TropicalCycloneTrain("tropicalCyclone")

    from dataset.image_dataloader import IMGDataloader
    from models.model_Baseline import BaselineNet

    model = BaselineNet(model_name="microsoft/aurora_small")
    model = model.to("cuda:0")

    data_loader = IMGDataloader(disaster="tropicalCyclone")
    criterion = nn.L1Loss()
    rmse_sur, rmse_atm, acc_sur, acc_atm = dict(), dict(), dict(), dict()
    train_data, META_INFO = data_loader.test_dataloader()
    for id, data in enumerate(train_data):
        batch = trainer.prepare_batch(data, META_INFO)
        # print(batch.metadata)
        logits = model(batch)

        logits_sur = [
            torch.tensor(logits.surf_vars[var].unsqueeze(1))
            for var in ["msl", "10u", "10v"]
        ]
        logits_sur = torch.cat(logits_sur, dim=1).squeeze(
            0
        )  # [1, 3, 1, 96, 96] -> [3, 1, 96, 96]

        logits_atm = [
            torch.tensor((logits.atmos_vars[var].unsqueeze(1)))
            for var in ["z", "u", "v"]
        ]
        logits_atm = torch.cat(logits_atm, dim=1).squeeze(0).squeeze(1)
        y_train = data["y"].to("cuda:0").squeeze(0)  # [..., :-1, :]
        y_atm_train = (
            data["y_upper"].to("cuda:0").squeeze(0).squeeze(1)
        )  # [..., :-1, :]

        if logits_atm.shape != y_atm_train.shape:
            print(f"resize from {logits_atm.shape[-2:]} to {y_atm_train.shape[-2:]}")
            logits_atm = nn.functional.interpolate(
                logits_atm, size=y_atm_train.shape[-2:], mode="bicubic"
            )
            logits_sur = nn.functional.interpolate(
                logits_sur, size=y_train.shape[-2:], mode="bicubic"
            )
        # print("logits_atm", logits_atm.shape)
        # print("logits_sur", logits_sur.shape)

        loss_sur = criterion(logits_sur, y_train)
        loss_atm = criterion(logits_atm, y_atm_train)

        loss = loss_sur + loss_atm
        # print(loss)
        # print("surface_loss", loss_sur)
        # print("upper level loss", loss_atm)
        stats = {}
        stats["means"] = torch.Tensor([[1.0122e05, -1.6686e00, 4.0252e-01]])
        stats["stds"] = torch.Tensor([[618.3453, 5.3051, 4.6408]])
        ec = (
            score_utils.unweighted_acc_torch(
                (
                    logits_sur.squeeze(1).unsqueeze(0)
                    - stats["means"].view(1, 3, 1, 1).to("cuda:0")
                )
                / stats["stds"].view(1, 3, 1, 1).to("cuda:0"),
                (
                    y_train.squeeze(1).unsqueeze(0)
                    - stats["means"].view(1, 3, 1, 1).to("cuda:0")
                )
                / stats["stds"].view(1, 3, 1, 1).to("cuda:0"),
            )
            .detach()
            .cpu()
            .numpy()[0]
        )

        if id % 50 == 0:
            target_test = y_train.detach().cpu().numpy()
            x = batch.surf_vars["10u"].detach().cpu().numpy()
            output_test = logits_sur.detach().cpu().numpy()
            min_c = np.amin(target_test[1, 0, :, :])
            max_c = np.amax(target_test[1, 0, :, :])

            fig, axes = plt.subplots(3, 1, figsize=(5, 15))
            im = axes[0].imshow(x[0, 0, :, :], vmin=min_c, vmax=max_c)
            plt.colorbar(im, ax=axes[0])
            axes[0].set_title("input")

            im = axes[1].imshow(target_test[1, 0, :, :])
            plt.colorbar(im, ax=axes[1])
            axes[1].set_title("target")

            im = axes[2].imshow(output_test[1, 0, :, :], vmin=min_c, vmax=max_c)
            plt.colorbar(im, ax=axes[2])
            axes[2].set_title("pred")
            plt.savefig(f"rollout_1_aurora_{id}_u10-2")
            plt.close(fig)
        break
