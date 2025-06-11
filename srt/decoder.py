import numpy as np
import torch
import torch.nn as nn

from srt.layers import RayEncoder, Transformer, PositionalEncoding


class RayPredictor(nn.Module):
    def __init__(self, num_att_blocks=2, pos_start_octave=0, out_dims=3,
                 z_dim=768, input_mlp=False, output_mlp=True):
        super().__init__()

        if input_mlp:  # Input MLP added with OSRT improvements
            self.input_mlp = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32))
        else:
            self.input_mlp = None

        self.query_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave,
                                        ray_octaves=15)
        self.transformer = Transformer(180, depth=num_att_blocks, heads=8, dim_head=z_dim // 8,
                                       mlp_dim=z_dim * 2, selfatt=False, kv_dim=z_dim)

        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(180, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims))
        else:
            self.output_mlp = None

    def forward(self, z, x, rays):
        """
        Args:
            z: scene encoding [num_patches, patch_dim]
            x: query camera positions [num_rays, 3]
            rays: query ray directions [num_rays, 3]
        """
        queries = self.query_encoder(x, rays) #(num_rays, 180)
        if self.input_mlp is not None:
            queries = self.input_mlp(queries)

        output = self.transformer(queries, z)
        if self.output_mlp is not None:
            output = self.output_mlp(output)
        return output


class GLNT_D(nn.Module):
    """ Scene Representation Transformer Decoder, as presented in the SRT paper at CVPR 2022"""
    def __init__(self, num_att_blocks=2, pos_start_octave=0):
        super().__init__()
        self.ray_predictor = RayPredictor(num_att_blocks=num_att_blocks,
                                          pos_start_octave=pos_start_octave,
                                          out_dims=64, z_dim=32,
                                          input_mlp=False, output_mlp=True)

    def forward(self, z, x, rays, **kwargs):
        """
        Args:
            z: scene encoding [h*w, n, c]
            x: query camera positions [num_rays, 3]
            rays: query ray directions [num_rays, 3]
        """
        z = z.flatten(0,1)
        output = self.ray_predictor(z, x, rays)
        return torch.sigmoid(output)
