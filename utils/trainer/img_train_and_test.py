import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils import logging_utils, score
import segmentation_models_pytorch as smp
from torchmetrics import AUROC, AveragePrecision, F1Score


class IMGTrain:
    def __init__(self, disaster):
        self.disaster = disaster
        self.loss_mapping = {
            "l1": nn.L1Loss(reduction="mean"),
            "dice": smp.losses.DiceLoss(mode="multiclass", from_logits=True),
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
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        # Loss function
        criterion = self.loss_mapping[loss]

        loss_list = []
        best_loss = np.inf

        for i in range(num_epochs):
            epoch_loss = 0.0
            start_time = time.time()
            for id, train_data in enumerate(train_loader):
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                # /with torch.cuda.amp.autocast():
                model.train()

                # Note the input and target need to be normalized (done within the function)
                # Call the model and get the output
                # x (b, w, h), y (b, w, h) ,mask (b, 3, w, h) , disno
                x = train_data["x"].to(device)  # (b, 1, w, h)
                # Move mask to the device and concatenate with x if present in train_data
                mask = train_data.get("mask")
                x_train = (
                    torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
                )
                y_train = train_data["y"].to(device)
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
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            end_time = time.time()
            logger.info(
                "Epoch {} : {:.3f} - Time cost: {:.3f}s".format(
                    i, epoch_loss, end_time - start_time
                )
            )
            # loss_list.append(epoch_loss)
            # To check the current learning rate
            # for param_group in optimizer.param_groups:
            #     print(f"Current Learning Rate: {param_group['lr']}")
            # Validate
            if i % 2 == 0:
                with torch.no_grad():
                    loss_val = 0
                    for id, val_data in enumerate(val_loader):
                        x = val_data["x"].to(device)
                        mask = val_data.get("mask")
                        x_val = (
                            torch.cat([x, mask.to(device)], dim=1)
                            if mask is not None
                            else x
                        )
                        y_val = val_data["y"].to(device)

                        logits_val = model(x_val)

                        loss = criterion(logits_val, y_val)
                        loss_val += loss.item()

                    loss_val /= len(val_loader)
                    lr_scheduler.step(loss_val)
                    logger.info("Val loss {} : {:.3f}".format(i, loss_val))
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
                        if i >= best_epoch + patience * 2:
                            break
            last_state = {key: value.cpu() for key, value in model.state_dict().items()}
            file_path = os.path.join(ckp_path, "last_model.pth")
            with open(file_path, "wb") as f:
                torch.save(last_state, f)
        return best_state, best_epoch, last_state, i


class ExtremeTemperatureTrain(IMGTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, test_loader, device, stats, save_path, model_id, **kwargs):
        rmse, acc = dict(), dict()
        criterion = nn.L1Loss()
        total_loss = 0
        # turn off gradient tracking for evaluation
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                x = test_data["x"].to(device)
                mask = test_data.get("mask")
                x_test = (
                    torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
                )
                y_test = test_data["y"].to(device)
                target_time = f"{test_data['disno'][0]}-{test_data['meta_info']['target_time'][0]}"

                model.eval()
                logits_test = model(x_test)  # (1, 1, 100, 100)
                loss = criterion(logits_test, y_test)

                # print("Test loss: {:.5f}".format(loss))
                # pred_test = pred_test.squeeze()
                # y_test = y_test.squeeze()
                acc[target_time] = (
                    score.unweighted_acc_torch(logits_test, y_test)
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
                # rmse
                csv_path = save_path / model_id / "csv"
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)

                output_test = logits_test * stats["std"] + stats["mean"]
                target_test = y_test * stats["std"] + stats["mean"]

                rmse[target_time] = (
                    score.unweighted_rmse_torch(output_test, target_test)
                    .detach()
                    .cpu()
                    .numpy()[0]
                )  # returns channel-wise score mean over w,h,b
                # visualize the last frame
                # put all tensors to cpu
                x = x * stats["std"] + stats[f"mean"]
                target_test = target_test.detach().cpu().numpy()
                x = x.detach().cpu().numpy()
                output_test = output_test.detach().cpu().numpy()
                fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                im = axes[0].imshow(x[0, 0])
                plt.colorbar(im, ax=axes[0])
                axes[0].set_title("input")

                im = axes[1].imshow(target_test[0, 0])
                plt.colorbar(im, ax=axes[1])
                axes[1].set_title("target")

                im = axes[2].imshow(output_test[0, 0])
                plt.colorbar(im, ax=axes[2])
                axes[2].set_title("pred")

                png_path = save_path / model_id / "png"
                if not os.path.exists(png_path):
                    os.makedirs(png_path)
                plt.savefig(f"{png_path}/test_pred_{target_time}.png")
                plt.close(fig)
                # Save rmses to csv

                total_loss += loss
            total_loss = total_loss / id
            logging_utils.save_errorScores(csv_path, acc, "acc")
            logging_utils.save_errorScores(csv_path, rmse, "rmse")

        return total_loss


class FireTrain(IMGTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, test_loader, device, stats, save_path, model_id, **kwargs):
        # turn off gradient tracking for evaluation
        f1 = dict()
        test_f1 = F1Score()
        criterion = smp.losses.DiceLoss(mode="multiclass")
        total_loss = 0
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                x = test_data["x"].to(device)
                mask = test_data.get("mask")
                x_test = (
                    torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
                )
                y_test = test_data["y"].to(device)
                model.eval()
                logits_test = model(x_test)
                loss = criterion(logits_test, y_test)
                pred_test = torch.nn.functional.softmax(logits_test, dim=1)
                pred_test = pred_test[:, -1]

                f1s = test_f1(
                    pred_test.detach().cpu().flatten(), y_test.detach().cpu().flatten()
                ).numpy()
                f1[test_data["meta_info"][0]] = f1s
                total_loss += loss
                # rmse
                csv_path = save_path / model_id / "csv"
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)
                mean = np.array([stats["means"][i] for i in [5, 3, 2]])
                std = np.array([stats["stds"][i] for i in [5, 3, 2]])

                target_test = y_test.detach().cpu().numpy()[0, 0]
                # only visualize the 1st item of a batch
                x_test = x_test.detach().cpu().numpy()[:, [5, 3, 2], :, :][0]
                output_test = pred_test.detach().cpu().numpy()[0]
                fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                normBackedData = x_test * std[:, None, None] + mean[:, None, None]
                normBackedData = (normBackedData - np.amin(normBackedData)) / (
                    np.amax(normBackedData) - np.amin(normBackedData)
                )
                im = axes[0].imshow(normBackedData.transpose((1, 2, 0)))
                plt.colorbar(im, ax=axes[0])
                axes[0].set_title("input")

                im = axes[1].imshow(target_test, vmin=0, vmax=1.0)
                plt.colorbar(im, ax=axes[1])
                axes[1].set_title("target")

                im = axes[2].imshow(output_test, vmin=0, vmax=1.0)
                plt.colorbar(im, ax=axes[2])
                axes[2].set_title("pred")

                png_path = save_path / model_id / "png"
                if not os.path.exists(png_path):
                    os.makedirs(png_path)
                st = test_data["meta_info"][0]
                plt.savefig(f"{png_path}/test_pred_{st}.png")
                plt.close()

                total_loss += loss
            total_loss = total_loss / id
            logging_utils.save_errorScores(csv_path, f1, "f1")

        return total_loss


class FloodTrain(IMGTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, test_loader, device, stats, save_path, model_id, **kwargs):
        # turn off gradient tracking for evaluation
        f1 = dict()
        test_f1 = F1Score(task="multiclass", num_classes=3, average=None)
        criterion = smp.losses.DiceLoss(mode="multiclass")
        total_loss = 0
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                x = test_data["x"].to(device)
                mask = test_data.get("mask")
                x_test = (
                    torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
                )
                y_test = test_data["y"].to(device)
                model.eval()
                logits_test = model(x_test)
                loss = criterion(logits_test, y_test)
                pred_test = torch.nn.functional.softmax(logits_test, dim=1)
                pred_test = torch.argmax(pred_test, dim=1)

                f1s = test_f1(
                    pred_test.detach().cpu().flatten(), y_test.detach().cpu().flatten()
                ).numpy()
                f1[test_data["meta_info"][0]] = f1s
                total_loss += loss

                csv_path = save_path / model_id / "csv"
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)
                mean = np.array([stats["means"][i] for i in [0, 1, 2]])
                std = np.array([stats["stds"][i] for i in [0, 1, 2]])

                target_test = y_test.detach().cpu().numpy()[0, 0]
                # only visualize the 1st item of a batch
                x_test = x_test.detach().cpu().numpy()[:, [0, 1, 2], :, :][0]
                output_test = pred_test.detach().cpu().numpy()[0]
                fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                normBackedData = x_test * std[:, None, None] + mean[:, None, None]
                normBackedData = (normBackedData - np.amin(normBackedData)) / (
                    np.amax(normBackedData) - np.amin(normBackedData)
                )
                im = axes[0].imshow(normBackedData.transpose((1, 2, 0)))
                plt.colorbar(im, ax=axes[0])
                axes[0].set_title("input")

                im = axes[1].imshow(target_test, vmin=0, vmax=2.0)
                plt.colorbar(im, ax=axes[1])
                axes[1].set_title("target")

                im = axes[2].imshow(output_test, vmin=0, vmax=2.0)
                plt.colorbar(im, ax=axes[2])
                axes[2].set_title("pred")

                png_path = save_path / model_id / "png"
                if not os.path.exists(png_path):
                    os.makedirs(png_path)
                st = test_data["meta_info"][0]
                plt.savefig(f"{png_path}/test_pred_{st}.png")
                plt.close()

                total_loss += loss
            total_loss = total_loss / id
            logging_utils.save_errorScores(csv_path, f1, "f1")

        return total_loss


if __name__ == "__main__":
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=1
    )

    # model = BaselineNet(input_dim=4, output_dim=1, model_name=model_name)
    model = model.to("cuda:0")
    batch = {"x_local": torch.randn((2, 3, 512, 512)).to("cuda:0")}

    output = model(batch)
    print(output)
