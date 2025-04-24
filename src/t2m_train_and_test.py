import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import seaborn as sns

from .models.model_components.aurora.aurora import Batch, Metadata
from .utils import logging_utils, score_utils

sys.path.insert(0, "/home/EarthExtreme-Bench")
import wandb


class AuroraTrain:
    def __init__(self, disaster):
        self.disaster = disaster
        self.loss_mapping = {
            "l1": nn.L1Loss(reduction="mean"),
            "l2": nn.MSELoss(reduction="mean"),
        }

    def prepare_batch(self, input, global_metas=None):
        local_metas = input["meta_info"]
        batch_size = input["x_u"].shape[0]
        spatial_res = local_metas["spatial_res"][0]
        # desired: (b, l, h, w), (h, w), (b, l, z, h, w)
        end_lon = local_metas["longitude"][0] + spatial_res * (input["x"].shape[-1])
        end_lon = 359.75 if end_lon >= 360 else end_lon

        batch = Batch(
            # (1, 1, 128,128)
            surf_vars={"2t": input["x_u"]},
            # (1, 3, 128, 128)
            static_vars={
                "lsm": input["mask_u"][0, 0],
                "slt": input["mask_u"][0, 1],
                "z": input["mask_u"][0, 2],
            },
            atmos_vars={"t": input["x_u"].unsqueeze(2)},
            metadata=Metadata(
                lat=torch.linspace(
                    local_metas["latitude"][0],
                    local_metas["latitude"][0] - spatial_res * (input["x_u"].shape[-2]),
                    input["x_u"].shape[-2],
                    dtype=torch.float32,
                ),
                lon=torch.linspace(
                    local_metas["longitude"][0],
                    end_lon,
                    input["x_u"].shape[-1],
                    dtype=torch.float32,
                ),
                time=tuple(
                    datetime.strptime(local_metas["input_time"][i], "%Y-%m-%d")
                    for i in range(batch_size)
                ),
                atmos_levels=(0,),
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
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False) #used in the paper

        # Loss function
        criterion = self.loss_mapping[loss]
        best_loss = np.inf  # np.inf
        best_state, last_state = None, None
        best_epoch = 0
        val_interval = 1
        train_loader, _ = data.train_dataloader()
        val_loader, _ = data.val_dataloader()
        """Training code"""

        for i in range(num_epochs):
            epoch_loss = 0.0

            for id, train_data in enumerate(train_loader):
                model.train()
                batch = self.prepare_batch(train_data)
                batch = batch.to(device)
                y_train = train_data["y_u"].to(device)
                # x(b, l1, w, h)

                output = model(batch)  # output.surf_vars["pcp"].requires_grad=True
                logits = (
                    output.surf_vars["2t"].clone().requires_grad_(True)
                )  # (b,t,h,w)

                loss = criterion(logits, y_train)
                optimizer.zero_grad()
                loss.backward()

                # Update model parameters with Adam optimizer
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                epoch_loss += loss.item()
            lr_scheduler.step()
            epoch_loss /= len(train_loader)

            wandb.log(
                {
                    "train loss": epoch_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            logger.info("Epoch {} : {:.3f}".format(i, epoch_loss))

            # Validate
            if i % val_interval == 0:
                with torch.no_grad():
                    loss_val = 0
                    for id, val_data in enumerate(val_loader):
                        batch = self.prepare_batch(val_data)
                        batch = batch.to(device)
                        y_val = val_data["y_u"].to(device)

                        output = model(batch)
                        logits = output.surf_vars["2t"].clone()  # (b,t,h,w)

                        loss = criterion(logits, y_val)
                        loss_val += loss.item()

                    loss_val /= len(val_loader)
                    logger.info("Val loss {} : {:.3f}".format(i, loss_val))
                    wandb.log({"validation loss": loss_val})
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_epoch = i
                        best_state = {
                            key: value.cpu()
                            for key, value in model.state_dict().items()
                        }
                        file_path = os.path.join(ckp_path, "best_model.pth")
                        with open(file_path, "wb") as f:
                            torch.save(best_state, f)
                            logger.info(
                                f"Saving the best model at epoch {best_epoch} to {file_path}...."
                            )
                    else:
                        if i >= best_epoch + patience * val_interval:
                            break
            last_state = {key: value.cpu() for key, value in model.state_dict().items()}
            file_path = os.path.join(ckp_path, "last_model.pth")
            with open(file_path, "wb") as f:
                torch.save(last_state, f)

        # best_epoch = i
        # file_path = os.path.join(ckp_path, "best_model.pth")
        # with open(file_path, "wb") as f:
        #     torch.save(last_state, f)
        #     logger.info(
        #         f"Saving the best model at epoch {best_epoch} to {file_path}...."
        #     )
        # return best_state, best_epoch, last_state, i
        return last_state, best_epoch, last_state, i


class ExtremeTemperatureTrain(AuroraTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, data, device, stats, save_path, model_id, **kwargs):
        test_loader, META_INFO = data.test_dataloader()
        rmse, rmse_normalized, cc = dict(), dict(), dict()
        criterion = nn.L1Loss()
        total_loss = 0
        # turn off gradient tracking for evaluation
        total_preds = []
        total_targets = []
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                batch = self.prepare_batch(test_data)
                batch = batch.to(device)
                y_test = test_data["y_u"].to(device)
                target_time = f"{test_data['disno'][0]}-{test_data['meta_info']['target_time'][0]}"

                model.eval()
                output_test = model(batch)  # (1, 1, 100, 100)
                logits_test = output_test.surf_vars["2t"]
                # if logits_test.size()[-1] != 224:
                #     logits_test = F.interpolate(
                #         logits_test,
                #         size=(224, 224),
                #         mode="bilinear",
                #     )
                #     y_test = F.interpolate(
                #         y_test,
                #         size=(224, 224),
                #         mode="bilinear",
                #     )

                loss = criterion(logits_test, y_test)

                # rmse
                csv_path = save_path / model_id / "csv"
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)
                # This computes the correlation coefficient between the normalized prediction and normalized target
                # It will return the same results as using acc(a*std, b*std)
                cc[target_time] = (
                    score_utils.unweighted_acc_torch(
                        logits_test - stats["mean"], y_test - stats["mean"]
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
                output_test = (logits_test - stats["mean"]) / stats["std"]
                target_test = (y_test - stats["mean"]) / stats["std"]

                total_preds.append(logits_test)
                total_targets.append(y_test)
                # compute the rmse on the normalized range
                rmse_normalized[target_time] = (
                    score_utils.unweighted_rmse_torch(output_test, target_test)
                    .detach()
                    .cpu()
                    .numpy()[0]
                )  # returns channel-wise score mean over w,h,b

                # compute the rmse on the raw range
                rmse[target_time] = (
                    score_utils.unweighted_rmse_torch(logits_test, y_test)
                    .detach()
                    .cpu()
                    .numpy()[0]
                )  # returns channel-wise score mean over w,h,b
                # visualize the last frame
                # put all tensors to cpu
                x = batch.surf_vars["2t"]
                # if x.size()[-1] != 224:
                #     x = F.interpolate(
                #         x,
                #         size=(224, 224),
                #         mode="bilinear",
                #     )
                y_test = y_test.detach().cpu().numpy()
                x = x.detach().cpu().numpy()
                logits_test = logits_test.detach().cpu().numpy()

                png_path = save_path / model_id / "png"
                if not os.path.exists(png_path):
                    os.makedirs(png_path)
                if id % 1 == 0:
                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    # im = axes[0].imshow(x[0, 0], cmap="RdBu")
                    # plt.colorbar(im, ax=axes[0], vmin=min_val, vmax=max_val)
                    # axes[0].set_title("input")
                    im = axes[0].imshow(logits_test[0, 0], cmap="RdBu_r")
                    cbar = plt.colorbar(im, orientation='vertical', pad=0.05, aspect=50)
                    cbar.set_label("K", fontsize=14)
                    axes[0].set_title("Prediction", fontsize=14)

                    im = axes[1].imshow(y_test[0, 0], cmap="RdBu_r")
                    plt.colorbar(im, ax=axes[1])
                    axes[1].set_title("Ground truth", fontsize=14)

                    im = axes[2].imshow(logits_test[0, 0]-y_test[0, 0], cmap="RdBu_r", vmin=-50, vmax=50)
                    plt.colorbar(im, ax=axes[2])
                    axes[2].set_title("Difference", fontsize=14)

                    plt.savefig(f"{png_path}/test_pred_{target_time}.png", dpi=300)
                    plt.close(fig)

                total_loss += loss
            total_loss = total_loss / id
            logging_utils.save_errorScores(csv_path, cc, "cc")
            logging_utils.save_errorScores(csv_path, rmse, "rmse")
            logging_utils.save_errorScores(csv_path, rmse_normalized, "nrmse")

            total_preds = torch.cat(total_preds, dim=0)
            total_targets = torch.cat(total_targets, dim=0)

            tq, tqe = score_utils.TQE(total_preds, total_targets)
            lq, lqe = score_utils.LQE(total_preds, total_targets)

            plt.figure(figsize=(10, 5))
            total_preds = total_preds.view(-1).detach().cpu().numpy()
            total_targets = total_targets.view(-1).detach().cpu().numpy()
            x_min = min(total_preds.min(), total_targets.min())
            x_max = max(total_preds.max(), total_targets.max())
            sns.histplot(total_preds, bins=100, kde=True, label='Predictions', alpha=0.6)
            sns.histplot(total_targets, bins=100, kde=True, label='Ground truth', alpha=0.6)
            plt.xlim(x_min, x_max)

            plt.xlabel("Temperature 2 meters", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.legend()
            plt.grid()
            plt.savefig(f"{png_path}/histogram.png", dpi=300)

        return {
            "total loss": total_loss,
            "top quantiles": tq,
            "top quantiles scores": tqe,
            "low quantiles": lq,
            "low quantiles scores": lqe,
        }


if __name__ == "__main__":
    pass
