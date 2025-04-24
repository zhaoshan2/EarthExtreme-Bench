import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import logging_utils, score_utils
import torch.nn.functional as F

sys.path.insert(0, "/home/EarthExtreme-Bench")
import wandb
from config.settings import settings


def _normback(img, mu, sigma):
    return img * sigma + mu


class SEQTrain:
    def __init__(self, disaster):
        self.disaster = disaster
        self.loss_mapping = {
            "l1": nn.L1Loss(reduction="mean"),
            "l2": nn.MSELoss(reduction="mean"),
        }

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
        start_time = time.time()
        best_loss = np.inf
        total_loss = 0.0
        iter_id = 0
        val_interval = 1000
        best_state, last_state = None, None
        best_epoch = 0
        train_loader = data.train_dataloader()
        if isinstance(train_loader, DataLoader):
            logger.info(f"length of training loader {len(train_loader)}")

        while iter_id < num_epochs:
            # sample a random minibatch
            try:
                train_data = next(train_loader)
            except StopIteration:
                break
            else:
                model.train()
                # vision models
                # x (l1, b, c, w, h) y (l2, b, c, w, h)
                x_train = train_data["x"].to(device)  # (b, 1, w, h)
                y_train = train_data["y"].to(device)
                # x(b, l1, w, h)

                if settings[self.disaster]["model"]["name"] == "ibm-nasa-geospatial/prithvi":
                    x_train = x_train.permute(1,2,0,3,4)  #( b, c, l1, w, h)
                else:
                    x_train = torch.transpose(x_train, 0, 1).squeeze(2)  # (b,l1, h, w)
                y_train = torch.transpose(y_train, 0, 1).squeeze(2)

                img_size = settings[self.disaster]["dataloader"]["img_size"]
                if x_train.size()[-1] != img_size:
                    if x_train.dim() == 5:
                        b, c, l, h, w = x_train.size()
                        x_train = F.interpolate(
                            x_train.view(b * c, l, h, w),
                            size=(img_size, img_size),
                            mode='nearest',
                        ).view(b, c, l, img_size, img_size)
                    else:
                        x_train = F.interpolate(
                            x_train,
                            size=(img_size, img_size),
                            mode='nearest',
                        )

                logits = model(
                    x_train
                )  # (upsampled) logits with the same w,h as inputs (b,c_out,w,h)
                if logits.size()[-2:] != y_train.size()[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=y_train.size()[-2:],
                        mode="nearest",
                    )
                loss = criterion(logits, y_train)
                # Call the backward algorithm and calculate the gratitude of parameters
                # scaler.scale(loss).backward()
                optimizer.zero_grad()
                loss.backward()

                # Update model parameters with Adam optimizer
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                lr_scheduler.step()
                total_loss += loss.item()
                iter_id += 1
                wandb.log(
                    {
                        "train loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                # Validate
                if iter_id % val_interval == 0:
                    logger.info("Iter {} : {:.3f}".format(iter_id, total_loss / val_interval))
                    total_loss = 0.0
                    val_loader = data.val_dataloader()
                    loss_val = 0
                    val_id = 0
                    while True:
                        try:
                            val_data = next(val_loader)
                            val_id += 1
                        except StopIteration:
                            break
                        with torch.no_grad():
                            x_val = val_data["x"].to(device)
                            y_val = val_data["y"].to(device)
                            if settings[self.disaster]["model"]["name"] == "ibm-nasa-geospatial/prithvi":
                                x_val = x_val.permute(1, 2, 0, 3, 4)
                            else:
                                x_val = torch.transpose(x_val, 0, 1).squeeze(2)
                            y_val = torch.transpose(y_val, 0, 1).squeeze(2)

                            if x_val.size()[-1] != img_size:
                                if x_val.dim()==5:
                                    b, c, l, h, w = x_val.size()
                                    x_val = F.interpolate(
                                        x_val.view(b * c, l, h, w),
                                        size=(img_size, img_size),
                                        mode='nearest',
                                    ).view(b, c, l, img_size, img_size)
                                else:
                                    x_val = F.interpolate(
                                                x_val,
                                                size=(img_size, img_size),
                                                mode="nearest",
                                            )
                            logits_val = model(x_val)
                            if logits_val.size()[-2:] != y_val.size()[-2:]:
                                logits_val = F.interpolate(
                                    logits_val,
                                    size=y_val.size()[-2:],
                                    mode="nearest",
                                )
                            loss = criterion(logits_val, y_val)
                            loss_val += loss.item()

                    loss_val /= val_id
                    logger.info("Val loss {} : {:.3f}".format(iter_id, loss_val))
                    wandb.log({"val loss": loss_val})
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_epoch = iter_id
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
                        if iter_id >= best_epoch + patience * val_interval:
                            break
                last_state = {
                    key: value.cpu() for key, value in model.state_dict().items()
                }
                file_path = os.path.join(ckp_path, "last_model.pth")
                with open(file_path, "wb") as f:
                    torch.save(last_state, f)

        logger.info(f"length of validation loader {len(val_loader)}")
        end_time = time.time()
        logger.info("Total training costs: {:.3f}s", end_time - start_time)
        return best_state, best_epoch, last_state, iter_id


class StormTrain(SEQTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, data, device, stats, save_path, model_id, seq_len):
        test_loader = data.test_dataloader()
        maes, mses = dict(), dict()
        criterion = nn.L1Loss()
        total_loss = 0
        thresholds = settings[self.disaster]["evaluation"]["thresholds"]
        thresholds = score_utils.rainfall_to_pixel(
            np.array(thresholds, dtype=np.float32)
        )  # thresholds in 0,1 range -> [0.328, 0.395, 0.462, 0.551, 0.618, 0.725]
        evaluator = score_utils.RadarEvaluation(seq_len=seq_len, thresholds=thresholds)
        scan_max = settings[self.disaster]["normalization"]["max"]
        scan_mean = settings[self.disaster]["normalization"][
            "pcp_mean"
        ]
        scan_sigma = settings[self.disaster]["normalization"][
            "pcp_std"
        ]
        # turn off gradient tracking for evaluation
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                x_test = test_data["x"].to(device)
                mask = test_data.get("mask")
                if settings[self.disaster]["model"]["name"] == "ibm-nasa-geospatial/prithvi":
                    x_test = x_test.permute(1, 2, 0, 3, 4)
                else:
                    x_test = torch.transpose(x_test, 0, 1).squeeze(2)
                y_test = test_data["y"].to(device)
                model.eval()
                img_size = settings[self.disaster]["dataloader"]["img_size"]

                if x_test.size()[-1] != img_size:
                    if x_test.dim() == 5:
                        b, c, l, h, w = x_test.size()
                        x_test = F.interpolate(
                            x_test.view(b * c, l, h, w),
                            size=(img_size, img_size),
                            mode='nearest',
                        ).view(b, c, l, img_size, img_size)
                    else:
                        x_test = F.interpolate(
                            x_test,
                            size=(img_size, img_size),
                            mode="nearest",
                        )
                logits_test = model(x_test)  # normalized

                if logits_test.size()[-2:] != y_test.size()[-2:]:
                    logits_test = F.interpolate(
                        logits_test,
                        size=y_test.size()[-2:],
                        mode="nearest",
                    )

                loss = criterion(logits_test, y_test.squeeze(2).transpose(0, 1))

                prediction = logits_test.transpose(0, 1).unsqueeze(2)
                prediction = torch.clamp(prediction, min=0)
                test_y_numpy = (
                    _normback(y_test.detach().cpu().numpy(), scan_mean, scan_sigma)
                    / scan_max
                )  # clipped to 0,1
                prediction_numpy = (
                    _normback(prediction.detach().cpu().numpy(), scan_mean, scan_sigma)
                    / scan_max
                )  # for evaluation, norm back to the original domain and then clip to [0,1]
                if mask is not None:
                    mse, mae = evaluator.update(
                        test_y_numpy, prediction_numpy, mask.cpu().numpy()
                    )
                else:
                    mse, mae = evaluator.update(test_y_numpy, prediction_numpy)

                datetime_seqs = test_data["meta_info"]["input_time"]
                for k in range(mae.shape[1]):
                    maes[datetime_seqs[k].strftime("%y-%m-%d %H:%M:%S")] = mae[:, k]
                    mses[datetime_seqs[k].strftime("%y-%m-%d %H:%M:%S")] = mse[:, k]
                # visualize the first batch and the first horizon
                if x_test.dim() == 5:
                    x_test = test_data["x"].to(device)
                    x_test = torch.transpose(x_test, 0, 1).squeeze(2)
                if x_test.size()[-1] != y_test.size()[-1]:
                    x_test = F.interpolate(
                        x_test,
                        size=y_test.size()[-2:],
                        mode="nearest",
                    )
                x = _normback(
                    x_test.transpose(0, 1).unsqueeze(2).detach().cpu().numpy(),
                    scan_mean,
                    scan_sigma,
                )
                mask = mask[0, 0, 0].numpy()
                if id % 10 == 0:
                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    im = axes[0].imshow(x[0, 0, -1] * mask)
                    plt.colorbar(im, ax=axes[0])
                    input_time = datetime_seqs[0] + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]*2
                    )
                    input_time = input_time.strftime("%y-%m-%dT%H%M")
                    axes[0].set_title(f"input_{input_time}")

                    im = axes[1].imshow(test_y_numpy[0, 0, 0] * scan_max * mask)
                    plt.colorbar(im, ax=axes[1])
                    target_time = datetime_seqs[0] + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]
                        * (settings[self.disaster]["dataloader"]["run_size"]-1)
                    )

                    axes[1].set_title(f"target_{target_time}")

                    im = axes[2].imshow(prediction_numpy[0, 0, 0] * scan_max * mask)
                    plt.colorbar(im, ax=axes[2])
                    axes[2].set_title(f"pred_{target_time}")

                    png_path = save_path / model_id / "png"
                    if not os.path.exists(png_path):
                        os.makedirs(png_path)
                    plt.savefig(f"{png_path}/test_pred_{input_time}.png")
                    plt.close(fig)
                total_loss += loss
            total_loss /= id
            csv_path = save_path / model_id / "csv"
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)
            test_pod, test_far, test_csi, test_hss, _, test_mse, test_mae, _ = (
                evaluator.calculate_stat()
            )
            logging_utils.save_errorScores(csv_path, maes, "nmaes")
            logging_utils.save_errorScores(csv_path, mses, "nmses")
        return {
            "loss": total_loss,
            "POD": test_pod,
            "FAR": test_far,
            "CSI": test_csi,
            "HSS": test_hss,
            "MSE": test_mse,
            "MAE": test_mae,
        }


class ExpcpTrain(SEQTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, data, device, stats, save_path, model_id, seq_len):
        test_loader = data.test_dataloader()
        maes, mses = dict(), dict()
        criterion = nn.L1Loss()
        total_loss = 0
        scan_max = settings[self.disaster]["normalization"]["max"]
        scan_mean = settings[self.disaster]["normalization"]["pcp_mean"]
        scan_sigma = settings[self.disaster]["normalization"]["pcp_std"]
        thresholds = settings[self.disaster]["evaluation"]["thresholds"]
        thresholds = [
            thresholds[i] / scan_max for i in range(len(thresholds))
        ]  # convert thresholds to [0,1]
        # turn off gradient tracking for evaluation
        evaluator = score_utils.RadarEvaluation(seq_len=seq_len, thresholds=thresholds)

        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                x_test = test_data["x"].to(device)
                if settings[self.disaster]["model"]["name"] == "ibm-nasa-geospatial/prithvi":
                    x_test = x_test.permute(1, 2, 0, 3, 4) # l, b, c, h, w -b, c, l, h, w
                else:
                    x_test = torch.transpose(x_test, 0, 1).squeeze(2)

                y_test = test_data["y"].to(device)
                model.eval()

                logits_test = model(x_test)
                loss = criterion(logits_test, y_test.squeeze(2).transpose(0, 1))
                # b, l , h, w
                prediction = logits_test.transpose(0, 1).unsqueeze(2) # l, b, 1, h, w
                prediction = torch.clamp(prediction, min=0)
                test_y_numpy = (
                    _normback(y_test.detach().cpu().numpy(), scan_mean, scan_sigma)
                    / scan_max
                )
                prediction_numpy = (
                    _normback(prediction.detach().cpu().numpy(), scan_mean, scan_sigma)
                    / scan_max
                )
                mse, mae = evaluator.update(test_y_numpy, prediction_numpy)

                datetime_seqs = test_data["meta_info"]["input_time"]
                for k in range(mae.shape[1]):
                    maes[datetime_seqs[k].strftime("%y-%m-%d %H:%M:%S")] = mae[:, k]
                    mses[datetime_seqs[k].strftime("%y-%m-%d %H:%M:%S")] = mse[:, k]
                # visualize the first batch and the first horizon

                x = _normback(
                    x_test.detach().cpu().numpy(), scan_mean, scan_sigma
                )
                if id % 1 == 0:
                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    im = axes[0].imshow(x[0, 0, -1])
                    plt.colorbar(im, ax=axes[0])
                    input_time = datetime_seqs[0] + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]*2
                    )
                    input_time = input_time.strftime("%y-%m-%dT%H%M")
                    axes[0].set_title(f"input_{input_time}")

                    im = axes[1].imshow(test_y_numpy[0, 0, 0] * scan_max)
                    plt.colorbar(im, ax=axes[1])
                    target_time = datetime_seqs[0] + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]
                        * (settings[self.disaster]["dataloader"]["run_size"]-1)
                    )
                    axes[1].set_title(f"target_{target_time}")

                    im = axes[2].imshow(prediction_numpy[0, 0, 0] * scan_max)
                    plt.colorbar(im, ax=axes[2])
                    axes[2].set_title(f"pred_{target_time}")

                    png_path = save_path / model_id / "png"
                    if not os.path.exists(png_path):
                        os.makedirs(png_path)
                    plt.savefig(f"{png_path}/test_pred_{input_time}.png")
                    plt.close(fig)
                total_loss += loss
            total_loss /= id
            csv_path = save_path / model_id / "csv"
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)
            test_pod, test_far, test_csi, test_hss, _, test_mse, test_mae, _ = (
                evaluator.calculate_stat()
            )
            logging_utils.save_errorScores(csv_path, maes, "maes")
            logging_utils.save_errorScores(csv_path, mses, "mses")
        return {
            "loss": total_loss,
            "POD": test_pod,
            "FAR": test_far,
            "CSI": test_csi,
            "HSS": test_hss,
            "MSE": test_mse,
            "MAE": test_mae,
        }


if __name__ == "__main__":
    from models.model_components.transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=20
    )

    # model = BaselineNet(input_dim=4, output_dim=1, model_name=model_name)
    model = model.to("cuda:0")
    batch = {"x_local": torch.randn((5, 3, 1, 512, 512)).to("cuda:0")}

    output = model(batch)
    print(output)
