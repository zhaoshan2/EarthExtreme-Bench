import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils import logging_utils, score


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
    **args,
):
    # training epoch
    epochs = num_epochs
    patience = patience
    """Training code"""
    # Prepare for the optimizer and scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False) #used in the paper

    # Loss function
    criterion = nn.L1Loss(reduction="mean")

    loss_list = []
    best_loss = np.inf

    for i in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        for id, train_data in enumerate(train_loader):

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            # /with torch.cuda.amp.autocast():
            model.train()

            # Note the input and target need to be normalized (done within the function)
            # Call the model and get the output
            # x (b, w, h), y (b, w, h) ,mask (b, 3, w, h) , disno
            x = train_data["x"].unsqueeze(1).to(device)  # (b, 1, w, h)
            mask = train_data["mask"][:, :2, :, :].to(device)  # (b, 3, w, h)
            x_train = torch.cat([x, mask], dim=1)  # (b, 4, w, h)
            y_train = train_data["y"].unsqueeze(1).to(device)
            output = model(x_train)  # (b,c_out,w,h)
            logits = output.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=y_train.shape[-2:], mode="bilinear", align_corners=False
            )

            # We use the MAE loss to train the model
            # Different weight can be applied for different fields if needed
            loss = criterion(upsampled_logits, y_train)
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
        print(
            "Epoch {} : {:.3f} - Time cost: {:.3f}s".format(
                i, epoch_loss, end_time - start_time
            )
        )
        # loss_list.append(epoch_loss)
        lr_scheduler.step(epoch_loss)

        # Validate
        if i % 5 == 0:
            with torch.no_grad():
                loss_val = 0
                for id, val_data in enumerate(val_loader):
                    x = val_data["x"].unsqueeze(1).to(device)
                    mask = val_data["mask"][:, :2, :, :].to(device)
                    x_val = torch.cat([x, mask], dim=1)
                    y_val = val_data["y"].unsqueeze(1).to(device)

                    outputs_val = model(x_val)
                    logits_val = outputs_val.logits
                    upsampled_logits_val = nn.functional.interpolate(
                        logits_val,
                        size=y_val.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss = criterion(upsampled_logits_val, y_val)
                    loss_val += loss.item()

                loss_val /= len(val_loader)
                print("Val loss {} : {:.3f}".format(i, loss_val))
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_epoch = i
                    best_state = {
                        key: value.cpu() for key, value in model.state_dict().items()
                    }
                    ckp_path = save_path / str(val_data["meta_info"]["disaster"][0])
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


def test(model, test_loader, device, stats, save_path, model_id):

    # turn off gradient tracking for evaluation
    rmse, acc = dict(), dict()
    criterion = nn.L1Loss()
    total_loss = 0
    with torch.no_grad():
        # iterate through test data
        for id, test_data in enumerate(test_loader):
            x = test_data["x"].unsqueeze(1).to(device)
            mask = test_data["mask"][:, :2, :, :].to(device)
            x_test = torch.cat([x, mask], dim=1)
            y_test = test_data["y"].unsqueeze(1).to(device)
            target_time = (
                f"{test_data['disno'][0]}-{test_data['meta_info']['target_time'][0]}"
            )

            model.eval()
            output_test = model(x_test)

            logits_test = output_test.logits
            upsampled_logits_test = nn.functional.interpolate(
                logits_test,
                size=y_test.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = criterion(upsampled_logits_test, y_test)

            # print("Test loss: {:.5f}".format(loss))
            # pred_test = pred_test.squeeze()
            # y_test = y_test.squeeze()
            acc[target_time] = (
                score.unweighted_acc_torch(upsampled_logits_test, y_test)
                .detach()
                .cpu()
                .numpy()[0]
            )
            # rmse
            disaster = test_data["meta_info"]["disaster"][0]
            csv_path = save_path / disaster / "csv" / model_id
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)

            output_test = (
                upsampled_logits_test * stats[f"{disaster}_std"]
                + stats[f"{disaster}_mean"]
            )
            target_test = y_test * stats[f"{disaster}_std"] + stats[f"{disaster}_mean"]

            rmse[target_time] = (
                score.unweighted_rmse_torch(output_test, target_test)
                .detach()
                .cpu()
                .numpy()[0]
            )  # returns channel-wise score mean over w,h,b
            # visualize the last frame
            # put all tensors to cpu
            x = x * stats[f"{disaster}_std"] + stats[f"{disaster}_mean"]
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

            png_path = save_path / disaster / "png" / model_id
            if not os.path.exists(png_path):
                os.makedirs(png_path)
            plt.savefig(f"{png_path}/test_pred_{target_time}.png")
            # Save rmses to csv

            total_loss += loss
        total_loss = total_loss / id
        logging_utils.save_errorScores(csv_path, acc, "acc")
        logging_utils.save_errorScores(csv_path, rmse, "rmse")

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
