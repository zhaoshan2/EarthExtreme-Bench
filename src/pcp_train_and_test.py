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
from .models.model_components.aurora.aurora import Batch, Metadata
from .utils import logging_utils, score_utils

sys.path.insert(0, "/home/EarthExtreme-Bench")
import wandb
from config.settings import settings


class SEQTrain:
    def __init__(self, disaster):
        self.disaster = disaster
        self.loss_mapping = {
            "l1": nn.L1Loss(reduction="mean"),
            "l2": nn.MSELoss(reduction="mean"),
        }

    def prepare_batch(self, input, global_metas=None):
        local_metas = input["meta_info"]
        batch_size = input["x"].shape[1]
        # desired: (b, l, h, w) now(l1, b, c, w, h)
        img_size = settings[self.disaster]["dataloader"]["img_size"]

        pcp_scaled = input["x"].transpose(0, 1).squeeze(2)
        mask_scaled = input["mask"][0]
        if pcp_scaled.shape[-1] != img_size:
            pcp_scaled = F.interpolate(
                                pcp_scaled,
                                size=(img_size, img_size),
                                mode='nearest',)
            mask_scaled = F.interpolate(
                                mask_scaled,
                                size=(img_size, img_size),
                                mode='nearest',)
        batch = Batch(
            surf_vars={"pcp": pcp_scaled},
            static_vars={"n": mask_scaled[0,0]},
            atmos_vars={"p": pcp_scaled.unsqueeze(2)},
            metadata=Metadata(
                lat=torch.linspace(
                    local_metas["latitude"][0],
                    local_metas["latitude"][0]
                    - settings[self.disaster]["spatial_res"] * (pcp_scaled.shape[-2]),
                    pcp_scaled.shape[-2],
                    dtype=torch.float32,
                ),
                lon=torch.linspace(
                    local_metas["longitude"][0],
                    local_metas["longitude"][0]
                    + settings[self.disaster]["spatial_res"] * (pcp_scaled.shape[-1]),
                    pcp_scaled.shape[-1],
                    dtype=torch.float32,
                ),
                time=tuple(local_metas["input_time"][i] for i in range(batch_size)),
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
        best_loss = np.inf  #
        total_loss = 0.0
        iter_id = 0  # 0
        best_state, last_state = None, None
        best_epoch = 0
        val_interval = 1000
        train_loader = data.train_dataloader()

        while iter_id < num_epochs:
            # sample a random minibatch
            try:
                train_data = next(train_loader)  # (25,4,1,480,480)
                batch = self.prepare_batch(train_data)
                batch = batch.to(device)
            except StopIteration:
                break
            else:
                optimizer.zero_grad()
                model.train()
                # input_sequence_length, batch_size, channels, width, height
                # x (l1, b, c, w, h) y (l2, b, c, w, h)
                y_train = train_data["y"].to(device)
                # x(b, l1, w, h)
                y_train = torch.transpose(y_train, 0, 1).squeeze(  # ( b, l2,  w, h)
                    2
                )  # (b,l,c w, h)->  (b,l, w, h)

                output = model(batch)  # output.surf_vars["pcp"].requires_grad=True
                logits = (
                    output.surf_vars["pcp"].clone().requires_grad_(True)
                )  # (b,t,h,w)
                if logits.size()[-2:] != y_train.size()[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=y_train.size()[-2:],
                        mode="nearest",
                    )
                # logits = torch.clamp(logits, min=0)
                loss = criterion(logits, y_train)
                # Call the backward algorithm and calculate the gratitude of parameters
                loss.backward()
                # Update model parameters with Adam optimizer
                optimizer.step()

                total_loss += loss.item()
                lr_scheduler.step()

                iter_id += 1
                wandb.log(
                    {
                        "train loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                # Validate
                if iter_id % val_interval == 0:
                    logger.info(
                        "iter {} : {:.3f}".format(iter_id, total_loss / val_interval)
                    )
                    total_loss = 0.0
                    val_loader = data.val_dataloader()
                    loss_val = 0.0
                    val_id = 0
                    while True:
                        try:
                            val_data = next(val_loader)
                            batch = self.prepare_batch(val_data)
                            batch = batch.to(device)
                            val_id += 1
                        except StopIteration:
                            break
                        with torch.no_grad():
                            output = model(
                                batch
                            )  # (upsampled) logits with the same w,h as inputs (b,c_out,w,h)
                            logits = output.surf_vars["pcp"]
                            y_val = val_data["y"].to(device)
                            y_val = torch.transpose(y_val, 0, 1).squeeze(2)
                            logits = torch.clamp(logits, min=0)
                            if logits.size()[-2:] != y_val.size()[-2:]:
                                logits = F.interpolate(
                                    logits,
                                    size=y_val.size()[-2:],
                                    mode="nearest",
                                )
                            loss = criterion(logits, y_val)
                            loss_val += loss.item()

                    loss_val /= val_id
                    # plot the last frame
                    logger.info("val loss {} : {:.3f}".format(iter_id, loss_val))
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

                # if iter_id % 100 == 0:
                with open(file_path, "wb") as f:
                    torch.save(last_state, f)

        logger.info(f"length of validation loader {len(val_loader)}")

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
        )  # [0.328, 0.395, 0.462, 0.551, 0.618, 0.725]

        # convert the thresholds(mm/h) to the pixel number using Z-R relationship
        evaluator = score_utils.RadarEvaluation(seq_len=seq_len, thresholds=thresholds)
        scan_max = settings[self.disaster]["normalization"]["max"]
        # turn off gradient tracking for evaluation
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                batch = self.prepare_batch(test_data)
                batch = batch.to(device)
                y_test = (
                    test_data["y"].to(device).transpose(0, 1).squeeze(2)
                )  # l, b, c, h, w
                model.eval()

                output = model(batch)
                logits_test = output.surf_vars["pcp"]  # b,l, h, w
                if logits_test.size()[-2:] != y_test.size()[-2:]:
                    logits_test = F.interpolate(
                        logits_test,
                        size=y_test.size()[-2:],
                        mode="bilinear",
                    )# b,l, 480, 480
                mask = test_data.get("mask")  # l, b, c, h, w

                loss = criterion(logits_test, y_test)  #  b,  h, w

                prediction = logits_test.unsqueeze(2).transpose(
                    0, 1
                )  # b, 1, 1, h, w - 1, b,1, h, w

                prediction = torch.clamp(prediction, min=0)
                # The evaluation takes all inputs should be between 0~1(pixel value) (pred, target, thresholds)
                test_y_numpy = (test_data["y"]).numpy() / scan_max  # l, b, c, h, w

                prediction_numpy = (
                    prediction.detach().cpu().numpy() / scan_max
                )  # l, b,c, h, w

                mse, mae = evaluator.update(
                    test_y_numpy,
                    prediction_numpy,
                    mask.cpu().numpy(),
                )

                datetime_seqs = test_data["meta_info"]["input_time"]
                for k in range(mae.shape[1]):
                    maes[datetime_seqs[k].strftime("%y-%m-%d %H:%M")] = mae[:, k]
                    mses[datetime_seqs[k].strftime("%y-%m-%d %H:%M")] = mse[:, k]

                # visualize the first batch and the first horizon convert the pixel number to rainfall intensity
                def better_vis(img):
                    # epsilon = 0.0001
                    # img = np.log(img + epsilon) - np.log(epsilon)
                    return img

                if id % 10 == 0:
                    x = batch.surf_vars["pcp"]
                    x = F.interpolate(
                        x,
                        size=(480, 480),
                        mode='nearest', ).detach().cpu().numpy()

                    mask = mask[0, 0, 0].numpy()
                    # b,l,h,w
                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    # s, l = 6, np.amax(better_vis(test_y_numpy[0, 0, 0]))
                    im = axes[0].imshow(
                        better_vis(x[0, -1] * mask)
                    )  # , vmin=s, vmax=l)
                    plt.colorbar(im, ax=axes[0])
                    input_time = datetime_seqs[0] + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]
                    )
                    input_time = input_time.strftime("%y-%m-%dT%H%M")
                    axes[0].set_title(f"input_{input_time}")

                    im = axes[1].imshow(better_vis(test_y_numpy[0, 0, 0] * scan_max * mask))  #
                    #     vmin=s,
                    #     vmax=l,
                    # )
                    plt.colorbar(im, ax=axes[1])
                    target_time = datetime_seqs[0] + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]
                        * settings[self.disaster]["dataloader"]["run_size"]
                    )
                    axes[1].set_title(f"target_{target_time}")

                    im = axes[2].imshow(better_vis(prediction_numpy[0, 0, 0] * scan_max * mask))  #
                    #     vmin=s,
                    #     vmax=l,
                    # )
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


class ExpcpTrain(StormTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, data, device, stats, save_path, model_id, seq_len):
        test_loader = data.test_dataloader()
        maes, mses = dict(), dict()
        criterion = nn.L1Loss()
        total_loss = 0
        scan_max = settings[self.disaster]["normalization"]["max"]
        thresholds = settings[self.disaster]["evaluation"]["thresholds"]
        thresholds = [
            thresholds[i] / scan_max for i in range(len(thresholds))
        ]

        evaluator = score_utils.RadarEvaluation(seq_len=seq_len, thresholds=thresholds)

        # turn off gradient tracking for evaluation
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                batch = self.prepare_batch(test_data)
                batch = batch.to(device)
                y_test = (
                    test_data["y"].to(device).transpose(0, 1).squeeze(2)
                )  # l, b, c, h, w
                model.eval()

                output = model(batch)
                logits_test = output.surf_vars["pcp"]  # b,l, h, w
                mask = test_data.get("mask")  # l, b, c, h, w

                loss = criterion(logits_test, y_test)  #  b,  h, w

                prediction = logits_test.unsqueeze(2).transpose(
                    0, 1
                )  # b, 1, 1, h, w - 1, b,1, h, w

                prediction = torch.clamp(prediction, min=0)
                test_y_numpy = test_data["y"].numpy()  # l, b, c, h, w
                prediction_numpy = prediction.detach().cpu().numpy()  # l, b,c, h, w

                if mask is not None:
                    mse, mae = evaluator.update(
                        test_y_numpy / scan_max,
                        prediction_numpy / scan_max,
                        mask.cpu().numpy(),
                    )
                else:
                    mse, mae = evaluator.update(
                        test_y_numpy / scan_max, prediction_numpy / scan_max
                    )

                datetime_seqs = test_data["meta_info"]["input_time"]
                for k in range(mae.shape[1]):
                    maes[datetime_seqs[k].strftime("%y-%m-%d %H:%M")] = mae[:, k]
                    mses[datetime_seqs[k].strftime("%y-%m-%d %H:%M")] = mse[:, k]
                # visualize the first batch and the first horizon
                if id % 10 == 0:
                    x = batch.surf_vars["pcp"].detach().cpu().numpy()  # b,l,h,w
                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    # s, l = np.percentile(test_y_numpy[0, 0, 0], 1), np.percentile(
                    #     test_y_numpy[0, 0, 0], 99.7
                    # )
                    s, l = np.amin(test_y_numpy[0, 0, 0]), np.amax(test_y_numpy[0, 0, 0])

                    im = axes[0].imshow(x[0, -1], vmin=s, vmax=l)
                    plt.colorbar(im, ax=axes[0])
                    input_time = datetime_seqs[0] + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]
                    )
                    input_time = input_time.strftime("%y-%m-%dT%H%M")
                    axes[0].set_title(f"input_{input_time}")

                    im = axes[1].imshow(test_y_numpy[0, 0, 0], vmin=s, vmax=l)
                    plt.colorbar(im, ax=axes[1])
                    target_time = datetime_seqs[0] + timedelta(
                        minutes=settings[self.disaster]["temporal_res"]
                        * (settings[self.disaster]["dataloader"]["run_size"]-1)
                    )
                    axes[1].set_title(f"target_{target_time}")
                    im = axes[2].imshow(prediction_numpy[0, 0, 0]*mask[0,0,0].detach().cpu().numpy(), vmin=s, vmax=l)
                    # im = axes[2].imshow(prediction_numpy[0, 0, 0], vmin=0.28)
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
