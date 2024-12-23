import torch
from torch import nn
import numpy as np
from models.model_utils import ResnetBlock, ConvBlock, MidLayer
from models.network_swinir import RSTB, PatchEmbed, PatchUnEmbed
import pdb

class HeadBS4(nn.Module):
    """HeadBS4 module - same as 2, without mid layer

    Args:
        channels (list): list of #channels in each upsampling layer
        pre_residuals (int, optional): number of residual blocks before upsampling. Default: 64
        down_conv_channels (list): list of #channels in each down_conv blocks
        up_residuals (int, optional): number of residual blocks in each upsampling module. Default: 0
    """
    def __init__(self, device, input_len, signals_count, channels,
            pre_residuals,
            pre_conv_channels,
            up_residuals,
            b_maxout,
            post_residuals,
            pow_2_channels,
            reduce_height,
            last_ch,
            activation,
            window_size,
            img_size,
            patch_size,
            depths,
            num_heads,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path_rate,
            norm_layer,
            downsample,
            resi_connection,
             bs_channels=2
          ):
        super(HeadBS4, self).__init__()
        
        self.device = device
        # update later to be from outside
        self.num_layers = len(depths)
        self.drop_rate = drop
        self.attn_drop_rate = attn_drop
        self.embed_dim = last_ch
        self.img_size=img_size
        self.patch_norm = True
        self.patch_size = patch_size
        self.depths = depths #[6, 6]
        self.window_size = window_size #8
        self.qkv_bias = qkv_bias # True
        self.qk_scale = qk_scale # None
        self.num_heads = num_heads # [2, 2]
        self.num_features = self.embed_dim
        self.linear = nn.Linear(last_ch, signals_count)
        self.act_fn = activation
        
        if norm_layer == True:
            self.norm_layer = nn.LayerNorm
        else:
            self.norm_layer = None
            
        if downsample == True:
            self.downsample = nn.Module
        else:
            self.downsample = None
            
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, 
            in_chans=self.embed_dim, embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        #pdb.set_trace()
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.resi_connection = resi_connection
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=self.img_size, 
            patch_size=self.patch_size, 
            in_chans=self.embed_dim, 
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)
        self.drop_path_rate = drop_path_rate
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        self.norm = self.norm_layer(self.num_features)
        
        # Initialize learnable output factor
        # self.f = torch.nn.Parameter(torch.ones(1))

        # Create pre_conv layer
        self.pre_conv = self._set_pre_conv_layers(bs_channels, 
                                                  pre_conv_channels, 
                                                  pre_residuals)
      
        self.transformer_layers = self._set_trsnsformer_layers(pre_conv_channels[-1])
        self.reduce_height = self._set_reduce_height(pre_conv_channels[-1], 
                                                     reduce_height)
                
        # Create post layer - only residuals, count set by input parameter
        self.post_conv = self._set_post_conv_layers(pre_conv_channels[-1], post_residuals)
        self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)


    def _set_trsnsformer_layers(self, last_pre_conv_ch):
        # build Residual Swin Transformer blocks (RSTB)

        layers = []
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        for i_layer in range(self.num_layers):
            layer = RSTB(dim=self.embed_dim,
                         input_resolution=(
                             self.patches_resolution[0],
                             self.patches_resolution[1]),
                             depth=self.depths[i_layer],
                             num_heads=self.num_heads[i_layer],
                             window_size=self.window_size,
                             qkv_bias=self.qkv_bias, 
                             qk_scale=self.qk_scale,
                             drop=self.drop_rate, 
                             attn_drop=self.attn_drop_rate,
                             drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],  # no impact on SR results
                             norm_layer=self.norm_layer,
                             downsample=self.downsample,
                             # use_checkpoint=use_checkpoint,
                             img_size=self.img_size,
                             patch_size=self.patch_size,
                             resi_connection=self.resi_connection
                         )
            layers.append(layer)

        return nn.Sequential(*layers)
     
    def forward_features(self, x):
        #pdb.set_trace()
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.transformer_layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x
        
    def _set_reduce_height(self, last_pre_conv_ch, reduce_height):
        c1 = last_pre_conv_ch
        # Create pre middle layer - reduce height
        reduce_height_layers = []
        cnt, k, s, add_conv_2= reduce_height

        for _ in range(cnt):
            reduce_height_layers.append(nn.Conv2d(in_channels=c1, out_channels=c1, 
                kernel_size=(k, 1), stride=(s, 1)))
        if add_conv_2:
            reduce_height_layers.append(nn.Conv2d(in_channels=c1, out_channels=c1, 
                kernel_size=(2, 1), stride=(2, 1)))
        return nn.Sequential(*reduce_height_layers)
        
    def _set_pre_conv_layers(self, bs_channels, pre_conv_channels, pre_residuals):
        pre_convs = []
        
        c0 = pre_conv_channels[0]
        
        # add first pre_conv layer: bs_channels-->pre_conv_channels[0]
        # add residuals after layer
        pre_convs.append(ConvBlock(bs_channels, c0, kernel_size=3, padding=1))
        # add resnets - no change in channels dim
        for _ in range(pre_residuals):
            pre_convs.append(ResnetBlock(c0, c0))
        
        # add additional pre_convs layer: pre_conv_channels[i]-->pre_conv_channels[i + 1]
        # add residuals after each layer
        # pre_conv_channels set by input parameter
        for i in range(len(pre_conv_channels) -1):
            in_c = pre_conv_channels[i]
            out_c = pre_conv_channels[i + 1]
            pre_convs.append(ResnetBlock(in_c, out_c))
            for _ in range(pre_residuals):
                pre_convs.append(ResnetBlock(out_c, out_c))
        return nn.Sequential(*pre_convs)

    def _set_post_conv_layers(self, c2, post_residuals):
        post_convs = []
        last_channels = int(c2)
        for i in range(post_residuals):
            post_convs.append(ResnetBlock(last_channels, last_channels, one_d=True, kernel_size=5))
        return nn.Sequential(*post_convs)
    
    def forward(self, x):
        """
        forward pass
        Args:
            x (Tensor): B x C x T # 100X100X2

        Returns:
            Tensor: B x C x (2^#channels * T) # 100X100X(2^#channels * 2)
        """
        x = self.pre_conv(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        # for BXCXHXW reduce dimension to BXCX1XW
        x = self.reduce_height(x)
        x = x.squeeze(2)
        x = self.post_conv(x)  
        x = self.linear(x.transpose(1, 2))
        x = self.act_fn(x).transpose(2, 1)

        return x

