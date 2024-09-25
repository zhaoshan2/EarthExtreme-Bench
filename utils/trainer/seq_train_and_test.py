import os
import time
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import logging_utils, score
import segmentation_models_pytorch as smp
from torchmetrics import AUROC, AveragePrecision, F1Score


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
        loss_list = []
        best_loss = np.inf
        total_loss = 0.0
        iter_id = 0
        best_state, last_state = None, None
        best_epoch = 0
        train_loader = data.train_dataloader()
        if isinstance(train_loader, DataLoader):
            logger.info(f"length of training loader {len(train_loader)}")

        while iter_id < num_epochs:
            # sample a random minibatch
            try:
                train_data = next(train_loader)  # (25,4,1,480,480)
            except StopIteration:
                break
            else:

                model.train()
                # x (l1, b, c, w, h) y (l2, b, c, w, h)
                x_train = train_data["x"].to(device)  # (b, 1, w, h)
                y_train = train_data["y"].to(device)
                # x(b, l1, w, h)
                x_train = torch.transpose(x_train, 0, 1).squeeze(2)
                y_train = torch.transpose(y_train, 0, 1).squeeze(2)
                logits = model(
                    x_train
                )  # (upsampled) logits with the same w,h as inputs (b,c_out,w,h)
                loss = criterion(logits, y_train)
                # Call the backward algorithm and calculate the gratitude of parameters
                # scaler.scale(loss).backward()
                optimizer.zero_grad()
                loss.backward()

                # Update model parameters with Adam optimizer
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                total_loss += loss.item()
                iter_id += 1
                # Validate
                if iter_id % 1000 == 0:
                    logger.info("Iter {} : {:.3f}".format(iter_id, total_loss / 1000))
                    total_loss = 0.0
                    val_loader = data.val_dataloader()
                    loss_val = 0
                    while True:
                        try:
                            val_data = next(val_loader)
                        except StopIteration:
                            break
                        with torch.no_grad():
                            x_val = val_data["x"].to(device)
                            y_val = val_data["y"].to(device)
                            x_val = torch.transpose(x_val, 0, 1).squeeze(2)
                            y_val = torch.transpose(y_val, 0, 1).squeeze(2)
                            logits_val = model(x_val)
                            loss = criterion(logits_val, y_val)
                            loss_val += loss.item()

                    loss_val /= len(val_loader)
                    lr_scheduler.step(loss_val)
                    logger.info("Val loss {} : {:.3f}".format(iter_id, loss_val))
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
                        if iter_id >= best_epoch + patience * 1000:
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

    def test(self, model, test_loader, device, stats, save_path, model_id, seq_len):
        maes, mses = dict(), dict()
        criterion = nn.L1Loss()
        total_loss = 0
        # turn off gradient tracking for evaluation
        evaluator = score.RadarEvaluation(seq_len=seq_len)
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                x_test = test_data["x"].to(device)
                mask = test_data.get("mask")
                x_test = torch.transpose(x_test, 0, 1).squeeze(2)
                y_test = test_data["y"].to(device)
                model.eval()

                logits_test = model(x_test)
                loss = criterion(logits_test, y_test.squeeze(2).transpose(0, 1))

                prediction = logits_test.transpose(0, 1).unsqueeze(2)
                test_y_numpy = y_test.detach().cpu().numpy()
                prediction_numpy = prediction.detach().cpu().numpy()
                if mask is not None:
                    mse, mae = evaluator.update(
                        test_y_numpy, prediction_numpy, mask.cpu().numpy()
                    )
                else:
                    mse, mae = evaluator.update(test_y_numpy, prediction_numpy)

                datetime_seqs = test_data["meta_info"]
                for k in range(mae.shape[1]):
                    maes[datetime_seqs[k].strftime("%y-%m-%d %H:%M:%S")] = mae[:, k]
                    mses[datetime_seqs[k].strftime("%y-%m-%d %H:%M:%S")] = mse[:, k]
                # visualize the first batch and the first horizon
                x = x_test.transpose(0, 1).unsqueeze(2).detach().cpu().numpy()
                fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                im = axes[0].imshow(x[0, 0, 0])
                plt.colorbar(im, ax=axes[0])
                input_time = datetime_seqs[0].strftime("%y-%m-%d %H:%M")
                axes[0].set_title(f"input_{input_time}")

                im = axes[1].imshow(test_y_numpy[0, 0, 0])
                plt.colorbar(im, ax=axes[1])
                target_time = datetime_seqs[0] + timedelta(minutes=5)
                axes[1].set_title(f"target_{target_time}")

                im = axes[2].imshow(prediction_numpy[0, 0, 0])
                plt.colorbar(im, ax=axes[2])
                axes[2].set_title(f"pred_{target_time}")

                png_path = save_path / model_id / "png"
                if not os.path.exists(png_path):
                    os.makedirs(png_path)
                plt.savefig(f"{png_path}/test_pred_{input_time}.png")
                plt.close(fig)
                total_loss += loss
            csv_path = save_path / model_id / "csv"
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)
            _, _, test_csi, test_hss, _, test_mse, test_mae, _ = (
                evaluator.calculate_stat()
            )
            total_loss = total_loss / id
            logging_utils.save_errorScores(csv_path, maes, "maes")
            logging_utils.save_errorScores(csv_path, mses, "mses")
        return {"CSI": test_csi, "HSS": test_hss, "MSE": test_mse, "MAE": test_mae}


class ExpcpTrain(StormTrain):
    def __init__(self, disaster):
        super().__init__(disaster)


if __name__ == "__main__":
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=20
    )

    # model = BaselineNet(input_dim=4, output_dim=1, model_name=model_name)
    model = model.to("cuda:0")
    batch = {"x_local": torch.randn((5, 3, 1, 512, 512)).to("cuda:0")}

    output = model(batch)
    print(output)
