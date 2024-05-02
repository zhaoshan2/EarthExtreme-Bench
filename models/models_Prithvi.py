# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import json
import os
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple

import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/EarthExtreme-Bench')
# from einops import rearrange
from utils.Prithvi_100M_config import model_args, data_args
from models.model_DecoderUtils import CoreDecoder
from utils import score
from utils import logging_utils
from pathlib import Path
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    """ Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            num_frames=3,
            tubelet_size=1,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (num_frames // tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_size[0], patch_size[1]), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x
    
class PrithviEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 num_frames=3, tubelet_size=1,
                 in_chans=3, embed_dim=1024, depth=24, num_heads=16, output_dim=1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280]
                 ):
        
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size,num_frames, tubelet_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]


        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # # remove cls token
        # x = x[:, 1:, :]
        return x


class Prithvi(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16,
                 num_frames=3, tubelet_size=1,
                 in_chans=3, embed_dim=1024, depth=24, num_heads=16, output_dim=1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280]
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.vit_encoder = PrithviEncoder(img_size=img_size, patch_size=patch_size,
                                          num_frames=num_frames, tubelet_size=tubelet_size,
                                          in_chans=in_chans, embed_dim=embed_dim, depth=depth, 
                                          num_heads=num_heads, output_dim=output_dim,
                                          mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                                          norm_pix_loss=norm_pix_loss,)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.output_dim = output_dim

        self.decoder_head = CoreDecoder(embedding_dim=embed_dim,
                                        output_dim=output_dim,
                                        depths=decoder_depths, 
                                        dims= decoder_dims,
                                        activation=decoder_activation,
                                        padding=decoder_padding, 
                                        norm=decoder_norm)
        
        self.decoder_downsample_block = nn.Identity()


    def reshape(self, x):
        # Separate channel axis
        N, L, D = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(N, D, int(L ** 0.5), int(L ** 0.5))

        return x

    def forward(self, x):
        x = x[:, :, None, :, :]
        x = self.vit_encoder(x)

        # remove cls token
        x = x[:, 1:, :]
        # reshape into 2d features
        x = self.reshape(x)
        x = self.decoder_downsample_block(x)
        x = self.decoder_head(x)
        return x

class PrithviClassifier(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16,
                 num_frames=3, tubelet_size=1,
                 in_chans=3, embed_dim=1024, depth=24, num_heads=16, output_dim=1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.vit_encoder = PrithviEncoder(img_size=img_size, patch_size=patch_size,
                                          num_frames=num_frames, tubelet_size=tubelet_size,
                                          in_chans=in_chans, embed_dim=embed_dim, depth=depth, 
                                          num_heads=num_heads, output_dim=output_dim,
                                          mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                                          norm_pix_loss=norm_pix_loss,)

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:

        self.classification_head = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=int(embed_dim/2)),
                                                 nn.LayerNorm(int(embed_dim/2)),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=int(embed_dim/2), out_features=output_dim)
                                                 )

    def forward(self, x):
        x = x[:, :, None, :, :]
        x = self.vit_encoder(x)
        # select cls token
        x = x[:, 0, :]
        x = self.classification_head(x)
        return x

def prithvi(checkpoint, output_dim=1, decoder_norm='batch', decoder_padding='same',
            decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280], freeze_body=True,
            classifier=False, inference=False):

    if classifier:
        model = PrithviClassifier(output_dim=output_dim,
                                  **model_args)

    else:
        model = Prithvi(output_dim=output_dim, decoder_norm=decoder_norm,  decoder_padding=decoder_padding,
                        decoder_activation=decoder_activation, decoder_depths=decoder_depths, decoder_dims=decoder_dims,
                        **model_args)

    if not inference:
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']

    # load pre-trained model
    msg = model.vit_encoder.load_state_dict(checkpoint, strict=False)
    print(msg)

    if freeze_body:
        for _, param in model.vit_encoder.named_parameters():
            param.requires_grad = False

    model.float()
    return model

def train(model, train_loader, val_loader, device, save_path: Path, **args):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 50], gamma=0.5)
    # training epoch
    epochs = 500
    patience = 20
    '''Training code'''
    # Prepare for the optimizer and scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False) #used in the paper

    # Loss function
    criterion = nn.L1Loss()

    loss_list = []
    best_loss = np.inf

    for i in range(epochs):
        epoch_loss = 0.0

        for id, train_data in enumerate(train_loader):

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            # /with torch.cuda.amp.autocast():
            model.train()

            # Note the input and target need to be normalized (done within the function)
            # Call the model and get the output
            # x (b, w, h), y (b, w, h) ,mask (b, 3, w, h) , disno
            x = train_data['x'].unsqueeze(1).to(device)  # (b, 1, w, h)
            mask = train_data['mask'].to(device) # (b, 3, w, h)
            x_train = torch.cat([x, mask, x, x ], dim=1)  # (b, 6, w, h)
            y_train = train_data['y'].unsqueeze(1).to(device)

            pred = model(x_train)  # (b,c_out,w,h)

            # We use the MAE loss to train the model
            # Different weight can be applied for different fields if needed
            loss = criterion(pred, y_train)
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
        print("Epoch {} : {:.3f}".format(i, epoch_loss))
        loss_list.append(epoch_loss)
        lr_scheduler.step()

        # Validate
        if i%10 == 0:
            with torch.no_grad():
                loss_val = 0
                for id, val_data in enumerate(val_loader):
                    x = val_data['x'].unsqueeze(1).to(device)
                    mask = val_data['mask'].to(device)
                    x_val = torch.cat([x, mask, x, x], dim=1)
                    y_val = val_data['y'].unsqueeze(1).to(device)

                    pred_val = model(x_val)
                    loss = criterion(pred_val, y_val)
                    loss_val += loss.item()
                loss_val /= len(val_loader)
                print("Val loss {} : {:.3f}".format(i, loss_val))
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_epoch = i
                    best_state = {key: value.cpu() for key, value in model.state_dict().items()}
                    ckp_path = save_path / str(val_data['meta_info']['disaster'][0])
                    if not os.path.exists(ckp_path):
                        os.mkdir(ckp_path)
                    file_path = os.path.join(ckp_path, "best_model.pth")
                    with open(file_path, 'wb') as f:
                        torch.save(best_state, f)
                else:
                    if i >= best_epoch + patience:
                        break
            # print("lr",lr_scheduler.get_last_lr()[0])
    return best_state

def test(model, test_loader, device, stats, save_path):

    # turn off gradient tracking for evaluation
    rmse, acc = dict(), dict()
    criterion = nn.L1Loss()
    with torch.no_grad():
        # iterate through test data
        for id, test_data in enumerate(test_loader):
            x = test_data['x'].unsqueeze(1).to(device)
            mask = test_data['mask'].to(device)
            x_test = torch.cat([x, mask, x, x], dim=1)
            y_test = test_data['y'].unsqueeze(1).to(device)
            target_time = f"{test_data['disno'][0]}-{test_data['meta_info']['target_time'][0]}"

            model.eval()
            pred_test = model(x_test)
            loss = criterion(pred_test, y_test)

            # print("Test loss: {:.5f}".format(loss))
            # pred_test = pred_test.squeeze()
            # y_test = y_test.squeeze()
            acc[target_time] = score.unweighted_acc_torch(pred_test, y_test).detach().cpu().numpy()[0]
            # rmse
            disaster= test_data['meta_info']['disaster'][0]
            csv_path = save_path / disaster / 'csv'
            if not os.path.exists(csv_path):
                os.mkdir(csv_path)

            output_test = pred_test * stats[f'{disaster}_std'] + stats[f'{disaster}_mean']
            target_test = y_test * stats[f'{disaster}_std'] + stats[f'{disaster}_mean']

            rmse[target_time] = score.unweighted_rmse_torch(output_test, target_test).detach().cpu().numpy()[0] #returns channel-wise score mean over w,h,b
        # Save rmses to csv
        logging_utils.save_errorScores(csv_path, acc, "acc")
        logging_utils.save_errorScores(csv_path, rmse, "rmse")

        # visualize the last frame
        # put all tensors to cpu
        x = x * stats[f'{disaster}_std'] + stats[f'{disaster}_mean']
        target_test = target_test.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        output_test = output_test.detach().cpu().numpy()
        fig, axes = plt.subplots(3, 1, figsize=(5, 15))
        im = axes[0].imshow(x[0, 0])
        plt.colorbar(im, ax=axes[0])
        axes[0].set_title("input")

        im = axes[1].imshow(target_test[0, 0])
        plt.colorbar(im, ax=axes[1])
        axes[1].set_title('target')

        im = axes[2].imshow(output_test[0, 0])
        plt.colorbar(im, ax=axes[2])
        axes[2].set_title('pred')

        png_path = save_path / disaster / 'png'
        if not os.path.exists(png_path):
            os.mkdir(png_path)
        plt.savefig(f'{png_path}/test_pred_pretrain_prithvi.png')

    return loss

if __name__ == '__main__':
    # main()
    # To do: problem with imcompetible keys
    """
    model = Prithvi(output_dim=1, **model_args)
    model = model.to(device)

    import utils.dataset.era5_extreme_temperature as da

    dataset = da.Era5HeatWave(horizon=28, chip_size=224)
    train_idx = 0
    test_idx = 183

    x = dataset[train_idx]['x'].unsqueeze(0).unsqueeze(1).to(device) # (1, 1, 128, 128)

    mask = dataset[train_idx]['mask'].unsqueeze(0).to(device)
    data = torch.cat([x,mask , x, x,], dim=1)# (1, 6, 224, 224)
    print("data", data.shape)

    data = (data - data_mean) /data_std
    # print(x.shape) # (w,h)
    y = dataset[train_idx]['y'].unsqueeze(0).unsqueeze(1).to(device)

    y = (y - 301.66) / 12.221
    best_model = train(model, data, y)
    # best_model = model

    x_test = dataset[test_idx]['x'].unsqueeze(0).unsqueeze(1).to(device) # (1, 1, 128, 128)

    mask_test = dataset[test_idx]['mask'].unsqueeze(0).to(device) #(1,3, 128, 128 )
    data_test = torch.cat([x_test, mask_test, x_test, x_test], dim=1)

    data_test = (data_test - data_mean) /data_std
    print("data_test", data_test.shape) # (w,h)
    y_test = dataset[test_idx]['y'].unsqueeze(0).unsqueeze(1).to(device)

    y_test = (y_test - 301.66) / 12.221

    pred_test = best_model(data_test)
    #pred_test = model(data_test)
    data_test = data_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    data_mean = 301.66
    data_std = 12.221
    vmin = 268
    vmax = 323
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    im = axes[0].imshow(data_test[0, 0]* data_std + data_mean)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("input")

    im = axes[1].imshow(y_test[0, 0]* data_std + data_mean)
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title('target')

    im = axes[2].imshow(pred_test[0, 0]* data_std + data_mean)
    plt.colorbar(im, ax=axes[2])
    axes[2].set_title('pred')
    plt.savefig('test_pred_nopretrain.png')

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CURR_FOLDER_PATH = Path(__file__).parent.parent  # "/home/EarthExtreme-Bench"
    SAVE_PATH = CURR_FOLDER_PATH / 'results' / 'Prithvi_100M'
    checkpoint = torch.load('/home/data_storage_home/data/disaster/pretrained_model/Prithvi_100M.pt')

    model = prithvi(checkpoint, output_dim=1, decoder_norm='batch', decoder_padding='same',
            decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280], freeze_body=True,
            classifier=False, inference=False)
    model.load_state_dict(torch.load(SAVE_PATH / 'heatwave' / 'best_model_200.pth'))

    model = model.to(device)

    import utils.dataset.era5_extreme_t2m_dataloader as ext

    heatwave = ext.HeateaveDataloader(batch_size=16,
                       num_workers=0,
                       pin_memory=False,
                       horizon=28,
                       chip_size=224,
                       val_ratio=0.5,
                       data_path='/home/EarthExtreme-Bench/data/weather',
                       persistent_workers=False)

    train_loader, records = heatwave.train_dataloader()
    val_loader, _ = heatwave.val_dataloader()

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    best_model_state_dict = train(model, train_loader, val_loader, device, save_path=SAVE_PATH)
    # best_model = model
    # checkpoint_trained = torch.load(SAVE_PATH / 'heatwave' / 'best_model.pth')
    msg = model.load_state_dict(best_model_state_dict)
    print(msg)

    test_loader, _ = heatwave.test_dataloader()
    _ = test(model, test_loader, device, stats=records.mean_std_dic, save_path=SAVE_PATH)


