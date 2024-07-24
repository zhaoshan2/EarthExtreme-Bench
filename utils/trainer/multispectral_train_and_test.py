import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics import AUROC, AveragePrecision, F1Score

from utils import logging_utils, score


# from torchmetrics.detection import IntersectionOverUnion
# from torchmetrics.classification import BinaryAccuracy
def train(
    model,
    train_loader,
    val_loader,
    device,
    save_path: Path,
    num_epochs,
    optimizer,
    lr_scheduler,
    patience=20,
    disaster="fire",
    **args,
):
    # training epoch
    epochs = num_epochs
    patience = patience
    """Training code"""
    # Prepare for the optimizer and scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False) #used in the paper

    # Loss function
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    best_loss = np.inf

    for i in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for id, train_data in enumerate(train_loader):

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            # with torch.cuda.amp.autocast():
            model.train()

            # Note the input and target need to be normalized (done within the function)
            x = train_data["x"].to(device)
            # if torch.isnan(x).any():
            #     raise ValueError(f"Input tensor {id} contains NaN values")
            mask = train_data["y"].to(device)  # (b, 1, w, h)
            x_train = x  # [:, [5, 3, 2], :, :]# (b, 6, w, h)
            y_train = mask.long()  # B,1,256,256

            logits = model(x_train)  # (b,c_out,w,h)
            # print("logits", logits.shape, logits)
            # B,2,256,256

            # upsampled_logits = nn.functional.interpolate(logits, size=y_train.shape[-2:], mode="bilinear", align_corners=False) # B,2,512,512

            # We use the MAE loss to train the model
            # Different weight can be applied for different fields if needed
            loss = criterion(logits, y_train.squeeze(1))

            # Call the backward algorithm and calculate the gratitude of parameters
            # scaler.scale(loss).backward()
            optimizer.zero_grad()
            loss.backward()

            # Update model parameters with Adam optimizer
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            epoch_loss += loss.item()
        end_time = time.time()
        epoch_loss /= len(train_loader)
        print(
            "Epoch {} : {:.3f} - Time cost: {:.3f}s".format(
                i, epoch_loss, end_time - start_time
            )
        )
        time_start = time.time()
        # loss_list.append(epoch_loss)
        lr_scheduler.step(epoch_loss)

        # Validate
        if i % 5 == 0:
            with torch.no_grad():
                loss_val = 0
                for id, val_data in enumerate(val_loader):
                    x = val_data["x"].to(device)  # (b, 1, w, h)
                    mask = val_data["y"].to(device)  # (b, 3, w, h)
                    x_val = x  # [:, [5, 3, 2], :, :] # (b, 4, w, h)
                    y_val = mask.long()
                    logits_val = model(x_val)

                    # upsampled_logits_val = nn.functional.interpolate(logits_val, size=y_val.shape[-2:], mode="bilinear", align_corners=False)
                    loss = criterion(logits_val, y_val.squeeze(1))
                    loss_val += loss.item()
                loss_val /= len(val_loader)
                print("Val loss {} : {:.3f}".format(i, loss_val))
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_epoch = i
                    best_state = {
                        key: value.cpu() for key, value in model.state_dict().items()
                    }
                    ckp_path = save_path / disaster
                    if not os.path.exists(ckp_path):
                        os.mkdir(ckp_path)
                    file_path = os.path.join(ckp_path, "best_model.pth")
                    with open(file_path, "wb") as f:
                        torch.save(best_state, f)
                        print(
                            f"Saving the best model at epoch {best_epoch} to {file_path}...."
                        )
                else:
                    if i >= best_epoch + patience:
                        break
            # print("lr",lr_scheduler.get_last_lr()[0])
    return best_state, best_epoch


def test(model, test_loader, device, save_path, disaster, stats=None, model_id=None):
    # For binary segmentation
    f1 = dict()
    test_f1 = F1Score()
    if disaster == "flood":
        test_f1 = F1Score(task="multiclass", num_classes=3, average=None)
    # turn off gradient tracking for evaluation
    criterion = smp.losses.DiceLoss(mode="multiclass")
    # test_auc = AUROC(pos_label=1, num_classes=2, compute_on_cpu=True)
    # test_auprc = AveragePrecision(pos_label=1, num_classes=1, compute_on_cpu=True)

    # test_acc = BinaryAccuracy()
    # test_iou = IntersectionOverUnion()

    total_loss = 0

    with torch.no_grad():
        # iterate through test data
        for id, test_data in enumerate(test_loader):
            x = test_data["x"].to(device)
            mask = test_data["y"].to(device)
            x_test = x  # [:, [5, 3, 2], :, :]
            y_test = mask.long()
            model.eval()
            logits_test = model(x_test)

            # upsampled_logits_test = nn.functional.interpolate(logits_test, size=y_test.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits_test, y_test.squeeze(1))
            pred_test = torch.nn.functional.softmax(logits_test, dim=1)  # [:, -1]
            # If two classes:
            if disaster == "fire":
                pred_test = pred_test[:, -1]  # [:, -1]
            elif disaster == "flood":
                # Convert probabilities to class map
                pred_test = torch.argmax(pred_test, dim=1)  # Shape: [1, 256, 256]
            else:
                raise ValueError(
                    f"Can't find matched evaluation map for disaster {disaster}"
                )

            # Convert to numpy array and squeeze to remove batch dimension
            # pred_test = pred_test.squeeze()  # Shape: [256, 256]
            # print("Test loss: {:.5f}".format(loss))

            # test_auc.update(pred_test.detach().cpu(), y_test.detach().cpu())
            # test_auprc.update(pred_test.detach().cpu().flatten(), y_test.detach().cpu().flatten())
            # test_f1.update(pred_test.detach().cpu().flatten(), y_test.detach().cpu().flatten())
            # test_data['id']: list of length 1
            # auc[test_data['id'][0]] = test_auc(pred_test.detach().cpu(), y_test.detach().cpu())
            # auprc[test_data['id'][0]]  = test_auprc(pred_test.detach().cpu().flatten(), y_test.detach().cpu().flatten())
            f1s = test_f1(
                pred_test.detach().cpu().flatten(), y_test.detach().cpu().flatten()
            ).numpy()
            f1[test_data["id"][0]] = f1s
            # f1_O[test_data['id'][0]] = f1s[1]
            # f1_U[test_data['id'][0]] = f1s[2]
            total_loss += loss
            # visualize the last frame
            # Mean and std of HLS Burned Fires
            means = np.array(
                [0.11944914225186566, 0.2323245113436119, 0.05889748132001316]
            )
            stds = np.array(
                [0.07241979477437814, 0.07791732423672691, 0.04004109844362779]
            )
            if disaster == "flood":
                means = np.array([0.23651549, 0.31761484, 0.18514981])
                stds = np.array([0.16280619, 0.20849304, 0.14008107])
            # target_test = y_test.detach().cpu().numpy()
            # # # x_test = x_test.detach().cpu().numpy()[:, [5, 3, 2], :, :]
            # x_test = x_test.detach().cpu().numpy()[:, [0,1,2], :, :]
            # output_test = pred_test.detach().cpu().numpy()
            # fig, axes = plt.subplots(3, 1, figsize=(5, 15))
            # normBackedData = x_test[0]*stds[:,None,None] + means[:,None,None]
            # normBackedData = (normBackedData - np.amin(normBackedData)) / (np.amax(normBackedData) - np.amin(normBackedData))
            # im = axes[0].imshow(normBackedData.transpose((1, 2, 0)))
            # plt.colorbar(im, ax=axes[0])
            # axes[0].set_title("input")
            #
            # im = axes[1].imshow(target_test[0,0], vmin=0, vmax=2.0)
            # plt.colorbar(im, ax=axes[1])
            # axes[1].set_title('target')
            #
            # im = axes[2].imshow(output_test[0], vmin=0, vmax=2.0)
            # plt.colorbar(im, ax=axes[2])
            # axes[2].set_title('pred')
            #
            # png_path = save_path / disaster / 'png' / model_id
            # if not os.path.exists(png_path):
            #     os.makedirs(png_path)
            # st =test_data['id'][0]
            # plt.savefig(f'{png_path}/test_pred_{st}.png')
            # plt.close()

        # Save scores to csv
        csv_path = save_path / disaster / "csv" / model_id
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        # logging_utils.save_errorScores(csv_path, auc, "auc")
        # logging_utils.save_errorScores(csv_path, auprc, "auprc")
        logging_utils.save_errorScores(csv_path, f1, "f1")

    return total_loss
