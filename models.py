""" Code for the main model variants. """
import torch
import torch.nn as nn

class R2P2_CNN(nn.Module):
    """
    R2P2 CNN Model
    """
    def __init__(self, in_channels=3):
        super(R2P2_CNN, self).__init__()
        self.conv_modules = nn.ModuleDict({
            'conv1': nn.Conv2d(in_channels, 32, 3, padding=0, dilation=1),
            'conv2': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv3': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv4': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv5': nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            'conv6': nn.Conv2d(32, 32, 3, padding=4, dilation=4),
            'conv7': nn.Conv2d(32, 32, 3, padding=8, dilation=8),
            'conv8': nn.Conv2d(32, 32, 3, padding=4, dilation=4),
            'conv9': nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            'conv10': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv11': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv12': nn.Conv2d(32, 32, 3, padding=0, dilation=1),
            'conv13': nn.Conv2d(32, 6, 1)
          })
        self.activations = nn.ModuleDict({
              'softplus': nn.Softplus(),
              'tanh': nn.Tanh()
          })

    def forward(self, x):
        '''
        input shape
        x: B X C X 64 X 64

        ouput shape
        final_output: B X 6 X 50 X 50
        hidden: [
            B X 32 X 62 X 62,
            B X 32 X 60 X 60,
            B X 32 X 58 X 58,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 56 X 56,
            B X 32 X 54 X 54,
            B X 32 X 52 X 52,
            B X 32 X 50 X 50,
            B X 6 X 50 X 50
          ] Conv1~Conv13 intermediate states
        '''
        return self.encode_scene(x)
    
    def encode_scene(self, x):
        conv_modules = self.conv_modules
        activations = self.activations

        hidden = []

        # Conv 1~10 with softplus
        for i in range(1, 11):
            x = conv_modules['conv{:d}'.format(i)](x)
            x = activations['softplus'](x)
            hidden.append(x)

        # Conv 11~12 with tanh
        for i in range(11, 13):
            x = conv_modules['conv{:d}'.format(i)](x)
            x = activations['tanh'](x)
            hidden.append(x)
          
        # Conv 13 (Linear 1X1)
        final_output = conv_modules['conv13'](x)
        hidden.append(final_output)

        return final_output, hidden

class R2P2_RNN(R2P2_CNN):
    """
    R2P2_RNN Model
    """
    def __init__(self,
                 context_encoder,
                 dynamic_decoder,
                 in_channels=3):

        super(R2P2_RNN, self).__init__(in_channels=in_channels)

        self.context_encoder = context_encoder
        self.dynamic_decoder = dynamic_decoder

    def forward(self, z, h_0, src_trajs, src_lens, decode_start_vel, decode_start_pos, scene_images):
        scene_encoding, _ = self.encode_scene(scene_images)
        
        src_trajs = src_trajs.permute(1, 0, 2) # Convert to (Time X Batch X Dim)
        context_encoding, _ = self.context_encoder(src_trajs, scene_encoding)

        x, mu, sigma = self.dynamic_decoder(z, h_0, context_encoding, decode_start_vel, decode_start_pos)
        
        return x, mu, sigma