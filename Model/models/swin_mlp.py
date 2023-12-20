# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    # Multilayer Perceptron class with two linear layers and a GELU activation in between
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # Initializes the MLP module. Allows for specifying the number of input, hidden, and output features.
        super().__init__()
        # Defaults hidden and output features to be the same as input features if not specified
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Activation function
        self.act = act_layer()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout layer
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Defines the forward pass of the MLP
        x = self.fc1(x)  # Apply first fully connected layer
        x = self.act(x)  # Apply activation function
        x = self.drop(x)  # Apply dropout
        x = self.fc2(x)  # Apply second fully connected layer
        x = self.drop(x)  # Apply dropout again
        return x

# window_partition: Partitions an image tensor into smaller window tensors

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

# window_reverse: Reconstructs the original image tensor from window tensors

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    r""" 
    Initializes the Swin MLP Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.dim = dim  # Number of input channels
        self.input_resolution = input_resolution  # Resolution of input feature map
        self.num_heads = num_heads  # Number of groups in group convolution
        self.window_size = window_size  # Size of the window for partitioning
        self.shift_size = shift_size  # Shift size for window partitioning
        self.mlp_ratio = mlp_ratio  # Ratio to determine the size of MLP's hidden layer

        # Adjust window size and shift size based on input resolution
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        # Padding for window partitioning
        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        # First normalization layer
        self.norm1 = norm_layer(dim)

        # Group convolution layer for multi-head MLP functionality
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        # Drop path layer for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Second normalization layer
        self.norm2 = norm_layer(dim)

        # MLP with specified hidden dimension
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Forward pass of the Swin MLP Block.

        Parameters:
        - x: Input tensor with shape (batch_size, length, channels).

        Returns:
        - Output tensor after processing by the Swin MLP Block.
        """
        
        # Extract height and width from input resolution
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        # Shortcut connection
        shortcut = x

        # Apply first normalization
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Apply shift and window partitioning
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Window/Shifted-Window Spatial MLP
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        # Merge window outputs
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # Reverse shift operation
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Apply MLP and add shortcut ("FFN")
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    # extra_repr: It provides a string representation for additional information about the module. 
    # When you print the module or its summary, this method is called to get a string 
    # that includes extra details about the module's properties
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    # flops: This method calculates the total number of floating-point operations required by the 
    # SwinMLPBlock. FLOPs are often used to estimate the computational cost of neural network models.

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W

        # Window/Shifted-Window Spatial MLP
        if self.shift_size > 0:
            nW = (H / self.window_size + 1) * (W / self.window_size + 1)
        else:
            nW = H * W / self.window_size / self.window_size
        flops += nW * self.dim * (self.window_size * self.window_size) * (self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

# PatchMerging: It is typically used in vision transformers to reduce the spatial dimensions 
# of the feature map

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    # PatchMerging.__init__: (Constructor) Initializes the PatchMerging layer.

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # Resolution of input feature map.
        self.dim = dim  # Number of input channels.
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # Linear layer for dimension reduction.
        self.norm = norm_layer(4 * dim)  # Normalization layer.

    # forward: Forward pass: Merges patches and reduces dimensions.

    def forward(self, x):
        H, W = self.input_resolution  # Height and Width of the input feature map.
        B, L, C = x.shape  # Batch size, Length of feature map, Number of channels.
        # Assertions to ensure correct input size.
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "x size must be even."

        # Reshape and merge patches.
        x = x.view(B, H, W, C)

        # Rearrange and concatenate patches.
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)  # Apply normalization.
        x = self.reduction(x)  # Reduce dimensions

        return x
    
    # extra_repr: String representation for additional information about the module.
    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"
    
    # flops: This method calculates the total number of floating-point operations required by this layer.
    
    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

# BasicLayer: This class represents a basic layer of a Swin MLP, 
# which contains several SwinMLPBlock modules and optionally a downsampling layer

class BasicLayer(nn.Module):
    """ A basic Swin MLP layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    # Constructor: Initializes the basic layer.

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim  # Number of input channels.
        self.input_resolution = input_resolution  # Input resolution.
        self.depth = depth  # Number of blocks in the layer.
        self.use_checkpoint = use_checkpoint  # Checkpointing for saving memory.

        # Building SwinMLPBlock modules.
        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        # Optional downsampling layer.
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    # Forward pass through the layer.

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    # extra_repr: String representation for additional information about the module.

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    
    # flops: Calculates the total number of floating-point operations (FLOPs) required by this layer.

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

# PatchEmbed: This class is used to split an image into patches and project them into an embedding space.

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    # PatchEmbed.__init__: (Constructor) Initializes the patch embedding layer.

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # Image size
        patch_size = to_2tuple(patch_size) # Patch size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] # Resolution of the grid of patches.
        self.img_size = img_size # Image size init 
        self.patch_size = patch_size # Patch size init
        self.patches_resolution = patches_resolution # Resolution init
        self.num_patches = patches_resolution[0] * patches_resolution[1] # Total number of patches.

        self.in_chans = in_chans  # Number of input channels.
        self.embed_dim = embed_dim  # Dimension of the patch embeddings.

        # Convolutional layer to project patches to embeddings.

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    # forward: Forward pass: Splits image into patches and projects them to embeddings.

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
    
    # flops: Calculates the total number of floating-point operations (FLOPs) required by this layer.

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

# SwinMLP: Defines the overall architecture of a Swin MLP model, 
# which is a type of neural network used for image classification tasks.

class SwinMLP(nn.Module):
    r""" Swin MLP

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    # SwinMLP.__init___: (Constructor) Initializes the Swin MLP model.

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        # Configuration of the model
        self.num_classes = num_classes  # Number of classes for the classification task.
        self.num_layers = len(depths)  # Total number of layers in the model.
        self.embed_dim = embed_dim  # Dimensionality of the patch embeddings.
        self.ape = ape  # Flag to add absolute position embedding.
        self.patch_norm = patch_norm  # Flag to apply normalization after patch embedding.
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # Final dimensionality of features.
        self.mlp_ratio = mlp_ratio  # Ratio of MLP hidden dimension to embedding dimension.

        # Patch Embedding: Converts input images to patch embeddings.
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute Position Embedding: Adds positional information to patch embeddings.
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        # Dropout layer applied after adding position embedding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic Depth: For regularization, a technique to drop layers randomly.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Building each layer in the Swin MLP.
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # Normalization layer applied to the final feature map.
        self.norm = norm_layer(self.num_features)

        # Adaptive average pooling layer.
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Final classification head: Linear layer for class predictions.
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights of the model.
        self.apply(self._init_weights)

    # _init_weights_: (Weight Initialization): Applies specific initializations to different 
    # types of layers.

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # no_weight_decay: Additional No Weight Decay Keywords: Specifies keywords for identifying 
    # parameters that should not have weight decay.

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    # no_weight_decay_keywords: Specifies keywords for identifying parameters that 
    # should not have weight decay.

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # forward_features: Processes input through patch embedding, position embedding, and subsequent 
    # layers.

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x
    
    # forward: Processes input through patch embedding, position embedding, and subsequent layers.

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    # flops: Calculates the total number of floating-point operations (FLOPs) required by this layer.

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        # Additional FLOPs for processing the final feature vector
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops