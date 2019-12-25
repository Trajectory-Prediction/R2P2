""" Code for the main model variants. """
import torch
import torch.nn as nn

class R2P2_RNN(nn.Module):
    """
    R2P2_RNN Model
    """
    def __init__(self,
                 scene_encoder,
                 static_encoder,
                 dynamic_generator):

        super(R2P2_RNN, self).__init__()

        self.scene_encoder = scene_encoder
        self.static_encoder = static_encoder
        self.dynamic_generator = dynamic_generator

    def forward(self, src_trajs, src_lens, decode_start_vel, decode_start_pos, scene_images):
        scene_encoding, _ = self.scene_encoder(scene_images)
        
        src_trajs = src_trajs.permute(1, 0, 2) # Convert to (Time X Batch X Dim)
        static_encoding, _ = self.static_encoder(src_trajs, scene_encoding)

        mu, sigma, x = self.dynamic_generator(static_encoding, decode_start_vel, decode_start_pos)
        return mu, sigma, x