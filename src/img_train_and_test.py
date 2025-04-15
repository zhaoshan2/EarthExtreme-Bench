import os
import time
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics import AUROC, AveragePrecision, JaccardIndex, F1Score, Accuracy
from .utils import logging_utils, score_utils
from sklearn.metrics import f1_score
import seaborn as sns

def my_f1(y_pred, y_true):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    print(TP, FP, FN, TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

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
        train_loader, _ = data.train_dataloader()
        val_loader, _ = data.val_dataloader()
        # Loss function
        criterion = self.loss_mapping[loss]

        best_loss = np.inf
        val_interval = 1

        for i in range(num_epochs):
            epoch_loss = 0.0

            for id, train_data in enumerate(train_loader):
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                # /with torch.cuda.amp.autocast():
                model.train()

                # Note the input and target need to be normalized (done within the function)
                x = train_data["x"].to(device)  # (b, 1, w, h)
                # Move mask to the device and concatenate with x if present in train_data
                mask = train_data.get("mask")
                coord_train = train_data.get("spatial_coords")
                x_train = (
                    torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
                )
                y_train = train_data["y"].to(device)
                if coord_train is not None and model.model_name == "ibm-nasa-geospatial/prithvi-2_upernet" :
                    logits = model(
                    (x_train,  None, coord_train)
                    )
                else:
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
                        x = val_data["x"].to(device)
                        mask = val_data.get("mask")
                        x_val = (
                            torch.cat([x, mask.to(device)], dim=1)
                            if mask is not None
                            else x
                        )
                        y_val = val_data["y"].to(device)

                        coord_val = val_data.get("coords")

                        if coord_val is not None  and model.model_name == "ibm-nasa-geospatial/prithvi-2_upernet":
                            logits_val = model(
                                (x_val, None, coord_val)
                            )
                        else:
                            logits_val = model(x_val)

                        if logits_val.size()[-2:] != y_val.size()[-2:]:
                            logits_val = F.interpolate(
                                logits_val,
                                size=y_val.size()[-2:],
                                mode="nearest",
                            )

                        loss = criterion(logits_val, y_val)
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

        return best_state, best_epoch, last_state, i


class ExtremeTemperatureTrain(IMGTrain):
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
                x = test_data["x"].to(device)
                mask = test_data.get("mask")
                x_test = (
                    torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
                )
                y_test = test_data["y"].to(device)
                coord_test = test_data.get("coords")

                target_time = f"{test_data['disno'][0]}-{test_data['meta_info']['target_time'][0]}"

                model.eval()
                if coord_test is not None:
                    logits_test = model(
                        (x_test, None, coord_test)
                    )
                else:
                    logits_test = model(x_test)  # (1, 1, 100, 100)
                loss = criterion(logits_test, y_test)

                # rmse
                csv_path = save_path / model_id / "csv"
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)
                # This computes the correlation coefficient between the normalized prediction and normalized target
                # It will return the same results as using acc(a*std, b*std)
                cc[target_time] = (
                    score_utils.unweighted_acc_torch(logits_test, y_test)
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
                # Compute the RMSE in the original data range
                output_test = logits_test * stats["std"] + stats["mean"]
                target_test = y_test * stats["std"] + stats["mean"]

                total_preds.append(output_test)
                total_targets.append(target_test)

                rmse[target_time] = (
                    score_utils.unweighted_rmse_torch(output_test, target_test)
                    .detach()
                    .cpu()
                    .numpy()[0]
                )  # returns channel-wise score mean over w,h,b

                # compute the rmse on the normalized space
                rmse_normalized[target_time] = (
                    score_utils.unweighted_rmse_torch(logits_test, y_test)
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

                png_path = save_path / model_id / "png"
                if not os.path.exists(png_path):
                    os.makedirs(png_path)
                if id % 1 == 0:
                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    max_val, min_val = np.amax(target_test[0, 0]), np.amin(target_test[0, 0])
                    # im = axes[0].imshow(x[0, 0], cmap="RdBu")
                    # plt.colorbar(im, ax=axes[0], vmin=min_val, vmax=max_val)
                    # axes[0].set_title("input")
                    im = axes[0].imshow(output_test[0, 0], cmap="RdBu_r")
                    cbar = plt.colorbar(im, orientation='vertical', pad=0.05, aspect=50)
                    cbar.set_label("K", fontsize=14)
                    axes[0].set_title("Prediction", fontsize=14)

                    im = axes[1].imshow(target_test[0, 0], cmap="RdBu_r")
                    plt.colorbar(im, ax=axes[1])
                    axes[1].set_title("Ground truth", fontsize=14)

                    im = axes[2].imshow(output_test[0, 0]-target_test[0, 0], cmap="RdBu_r", vmin=-50, vmax=50)
                    plt.colorbar(im, ax=axes[2])
                    axes[2].set_title("Difference", fontsize=14)

                    plt.savefig(f"{png_path}/test_pred_{target_time}.png", dpi=300)
                    plt.close(fig)
                # Save rmses to csv

                total_loss += loss
            total_loss = total_loss / id
            logging_utils.save_errorScores(csv_path, cc, "cc")
            logging_utils.save_errorScores(csv_path, rmse, "rmse")
            logging_utils.save_errorScores(csv_path, rmse_normalized, "nrmse")

            total_preds = torch.cat(total_preds, dim=0)
            total_targets = torch.cat(total_targets, dim=0)

            tq, tqe = score_utils.TQE(total_preds, total_targets)
            lq, lqe = score_utils.LQE(total_preds, total_targets)

            # Plot histograms
            plt.figure(figsize=(10, 5))
            total_preds = total_preds.view(-1).detach().cpu().numpy()
            total_targets = total_targets.view(-1).detach().cpu().numpy()
            x_min = min(total_preds.min(), total_targets.min())
            x_max = max(total_preds.max(), total_targets.max())
            sns.histplot(total_preds, bins=100, kde=True, label='Predictions', alpha=0.6)
            sns.histplot(total_targets, bins=100, kde=True, label='Ground truth', alpha=0.6)
            plt.xlim(x_min, x_max)
            # sns.hist(
            #     total_preds,
            #     bins=100,
            #     range=(np.amin(total_targets), np.amax(total_targets)),
            #     alpha=0.5,
            #     label="Predictions",
            #     color="red",
            # )
            # plt.hist(
            #     total_targets,
            #     bins=100,
            #     range=(np.amin(total_targets), np.amax(total_targets)),
            #     alpha=0.5,
            #     label="Ground truth",
            #     color="black",
            # )
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


class FireTrain(IMGTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, data, device, stats, save_path, model_id, **kwargs):
        # turn off gradient tracking for evaluation
        test_loader, META_INFO = data.test_dataloader()
        f1, IoU = dict(), dict()
        # Initialize F1 score metric from torchmetrics
        test_f1 = F1Score(num_classes=2, task="binary")
        test_IoU = JaccardIndex(task="multiclass", num_classes=2)
        test_macromIoU = JaccardIndex(task="multiclass", num_classes=2, average="macro")
        test_macromAcc = Accuracy(task="multiclass", num_classes=2, average="macro")
        criterion = smp.losses.DiceLoss(mode="multiclass")
        total_loss = 0
        total_pred, total_gt = [], []
        with torch.no_grad():
            # iterate through test data
            for id, test_data in enumerate(test_loader):
                x = test_data["x"].to(device)
                mask = test_data.get("mask")
                # To do: if the noise mask not impact the final performance, then remove it.
                noise_mask = test_data.get("noise_mask")
                x_test = (
                    torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
                )
                y_test = test_data["y"].to(device)  # b1hw
                coord_test = test_data.get("coords")
                model.eval()

                if coord_test is not None:
                    logits_test = model(
                        (x_test, None, coord_test)
                    )
                else:
                    logits_test = model(x_test)

                if logits_test.size()[-2:] != y_test.size()[-2:]:
                    logits_test = F.interpolate(
                        logits_test,
                        size=y_test.size()[-2:],
                        mode="nearest",
                    )
                loss = criterion(logits_test, y_test)

                pred_test = torch.nn.functional.softmax(logits_test, dim=1)  # b2hw
                # Returns the indices of the maximum value of all elements in the input tensor
                pred_test = torch.argmax(pred_test, dim=1).int()  # bhw

                total_pred.append(pred_test.flatten())
                total_gt.append(y_test.squeeze(1).flatten().int())


                f1s = test_f1(
                    pred_test.detach().cpu().flatten(),  # bhw
                    y_test.detach().cpu().squeeze(1).flatten(),
                ).numpy()

                ious = test_IoU(
                    pred_test.detach().cpu(),
                    y_test.detach().cpu().squeeze(1),
                ).numpy()

                f1[test_data["meta_info"][0]] = f1s
                IoU[test_data["meta_info"][0]] = ious

                csv_path = save_path / model_id / "csv"
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)

                if id % 10 == 0:
                    mean = np.array([stats["means"][i] for i in [5, 3, 2]])
                    std = np.array([stats["stds"][i] for i in [5, 3, 2]])

                    target_test = y_test.detach().cpu().numpy()[0, 0]
                    # only visualize the 1st item of a batch
                    x_test = x_test.detach().cpu().numpy()[:, [5, 3, 2], :, :][0]
                    output_test = pred_test.detach().cpu().numpy()[0]

                    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
                    cmap = plt.cm.get_cmap('viridis', 2)  # Discrete colormap with 3 categories
                    labels = ['No \nburned', 'Burned']

                    normBackedData = x_test * std[:, None, None] + mean[:, None, None]
                    normBackedData = (normBackedData - np.amin(normBackedData)) / (
                        np.amax(normBackedData) - np.amin(normBackedData)
                    )
                    im0= axes[0].imshow(normBackedData.transpose((1, 2, 0)))
                    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', pad=0.05)
                    cbar0.set_label('Normalized reflectance', fontsize=14)
                    axes[0].set_title("Input", fontsize=14)
                    axes[0].axis('off')


                    im1 = axes[1].imshow(target_test, vmin=0, vmax=1.0, cmap=cmap)
                    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', pad=0.05)
                    cbar1.set_ticks([0, 1])  # Explicitly set the tick locations
                    cbar1.set_ticklabels(labels, fontsize=14)  # Set the tick labels
                    for tick in cbar1.ax.get_yticklabels():  # Rotate tick labels
                        tick.set_rotation(90)
                    # cbar1.set_label('Categories', fontsize=14)
                    axes[1].set_title("Prediction", fontsize=14)
                    axes[1].axis('off')

                    im2 = axes[2].imshow(output_test, vmin=0, vmax=1.0, cmap=cmap)
                    cbar2 = fig.colorbar(im2, ax=axes[2], orientation='vertical', pad=0.05)
                    cbar2.set_ticks([0, 1])  # Explicitly set the tick locations
                    cbar2.set_ticklabels(labels, fontsize=14)  # Set the tick labels
                    # for tick in cbar2.ax.get_yticklabels():  # Rotate tick labels
                    #     tick.set_rotation(90)
                    # cbar2.set_label('Categories', fontsize=14)
                    axes[2].set_title("Prediction", fontsize=14)
                    axes[2].axis('off')


                    png_path = save_path / model_id / "png"
                    if not os.path.exists(png_path):
                        os.makedirs(png_path)
                    st = test_data["meta_info"][0]
                    plt.savefig(f"{png_path}/test_pred_{st}.png")
                    plt.close()

                total_loss += loss
            total_loss = total_loss / id

            final_pred = torch.cat(total_pred, dim=0)
            final_gt = torch.cat(total_gt, dim=0)

            f1_unweighted = test_f1(
                final_pred.detach().cpu(),
                final_gt.detach().cpu(),
            )

            iou_unweighted = test_IoU(
                final_pred.detach().cpu(), final_gt.detach().cpu()
            )
            miou_macro = test_macromIoU(
                final_pred.detach().cpu(), final_gt.detach().cpu()
            )
            macc_macro = test_macromAcc(
                final_pred.detach().cpu(), final_gt.detach().cpu()
            )
            logging_utils.save_errorScores(csv_path, f1, "f1")
            logging_utils.save_errorScores(csv_path, IoU, "IoU")

        return {
            "total_loss": total_loss,
            "f1": f1_unweighted,
            "iou": iou_unweighted,
            "macro_mIoU": miou_macro,
            "macro_mAcc": macc_macro,
        }


class FloodTrain(IMGTrain):
    def __init__(self, disaster):
        super().__init__(disaster)

    def test(self, model, data, device, stats, save_path, model_id, **kwargs):
        test_loader, META_INFO = data.test_dataloader()
        # turn off gradient tracking for evaluation
        f1, IoU = dict(), dict()
        test_f1 = F1Score(
            task="multiclass", num_classes=3, average=None
        )  # [f1_cls0, f1_cls1, f1_cls2]
        test_f1_weighted = F1Score(
            task="multiclass", num_classes=3, average="weighted", ignore_index=0
        )  # single number
        test_f1_macro = F1Score(
            task="multiclass", num_classes=3, average="macro", ignore_index=0
        )  # single number
        test_IoU = JaccardIndex(task="multiclass", num_classes=3, average=None)
        test_IoU_macro = JaccardIndex(
            task="multiclass", num_classes=3, average="macro", ignore_index=0
        )  # single number
        test_IoU_weighted = JaccardIndex(
            task="multiclass", num_classes=3, average="weighted", ignore_index=0
        )  # single number

        criterion = smp.losses.DiceLoss(mode="multiclass")
        total_loss = 0
        total_pred, total_gt = [], []
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
                coord_test = test_data.get("coords")

                if coord_test is not None and model.model_name == "ibm-nasa-geospatial/prithvi-2_upernet":
                    logits_test = model(
                        (x_test, None, coord_test)
                    )
                else:
                    logits_test = model(x_test)

                if logits_test.size()[-2:] != y_test.size()[-2:]:
                    logits_test = F.interpolate(
                        logits_test,
                        size=y_test.size()[-2:],
                        mode="nearest",
                    )
                loss = criterion(logits_test, y_test)
                pred_test = torch.nn.functional.softmax(logits_test, dim=1)
                pred_test = torch.argmax(pred_test, dim=1).int()

                total_pred.append(pred_test.flatten())
                total_gt.append(y_test.squeeze(1).int().flatten())
                # f1 for each sample [f1_cls0, f1_cls1, f1_cls2]
                f1s = test_f1(
                    pred_test.detach().cpu().flatten(),
                    y_test.squeeze(1).detach().cpu().flatten(),
                ).numpy()
                # IoU for each sample
                ious = test_IoU(
                    pred_test.detach().cpu().flatten(),
                    y_test.squeeze(1).detach().cpu().flatten(),
                ).numpy()

                f1[test_data["meta_info"][0]] = f1s
                IoU[test_data["meta_info"][0]] = ious

                csv_path = save_path / model_id / "csv"
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)
                bands = [7,1,3] # use 8,2,4 bands for flood visualization

                if id % 100 == 0:
                    mean = np.array([stats["means"][i] for i in bands])
                    std = np.array([stats["stds"][i] for i in bands])

                    target_test = y_test.detach().cpu().numpy()[0, 0]
                    # only visualize the 1st item of a batch
                    x_test = x_test.detach().cpu().numpy()[:, bands, :, :][0]
                    output_test = pred_test.detach().cpu().numpy()[0]

                    # Define the colormap and labels
                    cmap = plt.cm.get_cmap('viridis', 3)  # Discrete colormap with 3 categories
                    labels = ['No flood', 'Open flood', 'Urban flood']

                    # Create the figure and axes
                    fig, axes = plt.subplots(3, 1, figsize=(8, 15), constrained_layout=True)

                    # Normalize the input data
                    normBackedData = x_test * std[:, None, None] + mean[:, None, None]
                    normBackedData = (normBackedData - np.amin(normBackedData)) / (
                            np.amax(normBackedData) - np.amin(normBackedData)
                    )

                    # Plot the input image
                    im0 = axes[0].imshow(normBackedData.transpose((1, 2, 0)))
                    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', pad=0.05)
                    cbar0.set_label('Normalized input', fontsize=14)
                    # axes[0].set_title("Input Image", fontsize=14)
                    axes[0].axis('off')

                    # Plot the target image
                    im1 = axes[1].imshow(target_test, cmap=cmap, vmin=0, vmax=2.0)
                    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', pad=0.05)
                    cbar1.set_ticks([0, 1, 2])  # Explicitly set the tick locations
                    cbar1.set_ticklabels(labels, fontsize=14)  # Set the tick labels
                    for tick in cbar1.ax.get_yticklabels():  # Rotate tick labels
                        tick.set_rotation(90)
                    cbar1.set_label('Categories', fontsize=14)
                    # axes[1].set_title("Target Image", fontsize=14)
                    axes[1].axis('off')

                    # Plot the predicted image
                    im2 = axes[2].imshow(output_test, cmap=cmap, vmin=0, vmax=2.0)
                    cbar2 = fig.colorbar(im2, ax=axes[2], orientation='vertical', pad=0.05)
                    cbar2.set_ticks([0, 1, 2])  # Explicitly set the tick locations
                    cbar2.set_ticklabels(labels, fontsize=14)  # Set the tick labels
                    for tick in cbar2.ax.get_yticklabels():  # Rotate tick labels
                        tick.set_rotation(90)
                    cbar2.set_label('Categories', fontsize=14)
                    # axes[2].set_title("Predicted Image", fontsize=14)
                    axes[2].axis('off')

                    png_path = save_path / model_id / "png"
                    if not os.path.exists(png_path):
                        os.makedirs(png_path)
                    st = test_data["meta_info"][0]
                    plt.savefig(f"{png_path}/test_pred_{st}.png", dpi=200)
                    plt.close()

                total_loss += loss
            total_loss = total_loss / id

            logging_utils.save_errorScores(csv_path, f1, "f1")
            logging_utils.save_errorScores(csv_path, IoU, "IoU")

            final_pred = torch.cat(total_pred, dim=0)
            final_gt = torch.cat(total_gt, dim=0)
            # pixel-wise weighted average f1
            f1_weighted = test_f1_weighted(
                final_pred.detach().cpu(), final_gt.detach().cpu()
            ).numpy()
            # pixel-wise marco average f1
            f1_macro = test_f1_macro(
                final_pred.detach().cpu(), final_gt.detach().cpu()
            ).numpy()
            # pixel-wise f1
            f1_unweighted = test_f1(
                final_pred.detach().cpu(),
                final_gt.detach().cpu(),
            ).numpy()

            # pixel-wise IoU
            iou_unweighted = test_IoU(
                final_pred.detach().cpu(), final_gt.detach().cpu()
            ).numpy()
            # pixel-wise weighted average IoU
            iou_weighted = test_IoU_weighted(
                final_pred.detach().cpu(), final_gt.detach().cpu()
            ).numpy()
            # pixel-wise macro average IoU
            iou_macro = test_IoU_macro(
                final_pred.detach().cpu(), final_gt.detach().cpu()
            ).numpy()
        return {
            "total_loss": total_loss,
            "f1": f1_unweighted,
            "mF1_weighted": f1_weighted,
            "mF1_macro": f1_macro,
            "IoU": iou_unweighted,
            "mIoU_weighted": iou_weighted,
            "mIoU_macro": iou_macro,
        }


if __name__ == "__main__":
    from models.model_components.transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=1
    )

    # model = BaselineNet(input_dim=4, output_dim=1, model_name=model_name)
    model = model.to("cuda:0")
    batch = {"x_local": torch.randn((2, 3, 512, 512)).to("cuda:0")}

    output = model(batch)
    print(output)
