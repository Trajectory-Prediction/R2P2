import time

import torch
import torch.nn as nn

import pdb

class ContextEncoder(nn.Module):
  """
  Context Encoder for R2P2 RNN
  """
  def __init__(self, encoding_size=50, hidden_size=150, scene_size=50, scene_channels=6):
    """
    Default ketword params are from the original settings in R2P2 paper.
    encoding_size: size of final encoding vector to return.
    hidden_size: hidden state size for the motion states encoder (GRU cell).
    scene_size: width and height of the scene encoding (Currently int value since we assume retangular shape).
    scene_channels: # channels of the scene encdoing.
    """
    super(ContextEncoder, self).__init__()
    self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, num_layers=1)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_size + scene_size * scene_size * scene_channels, encoding_size),
        nn.Softplus(),
        nn.Linear(encoding_size, encoding_size),
        nn.Softplus()
      )
  
  def forward(self, x, scene):
    '''
    input shape
    x: 20 X B X 2 
    scene: B X 6 X 50 X 50
    
    ouput shape
    final_output: B X 50
    hidden: [
        B X 150,
        B X 150 + 6 * 50 * 50
        B X 50
      ]
    '''
    # Detect dynamic batch size
    batch_size = scene.size(0)

    hidden = []

    motion_encoding, _ = self.gru(x)
    motion_encoding = motion_encoding[-1] # Need the last one
    hidden.append(motion_encoding)

    # Flatten the scene encoding then concat with the motion encoding
    scene_vec = scene.reshape((batch_size, -1))
    concat_encoding = torch.cat((motion_encoding, scene_vec), dim=1)
    hidden.append(concat_encoding)

    # 2-layer MLP
    final_output = self.mlp(concat_encoding)
    hidden.append(final_output)

    return final_output, hidden

class DynamicDecoder(nn.Module):
    """
    Dynamic Decoder for R2P2 RNN
    """
    def __init__(self):
        super(DynamicDecoder, self).__init__()
        self.gru = nn.GRU(input_size=30*2, hidden_size=150, num_layers=1)
        self.mlp = nn.Sequential(
            nn.Linear(150+50, 50),
            nn.Softplus(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 6)
        )

    def forward(self, z, h_0, static, init_velocity, init_position):
        '''
        input shape
        z: B X 60
        h_0: B X 150
        static: B X 50
        init_velocity: B X 2
        init_position: B X 2
        
        ouput shape
        x : B X 30 X 2
        mu: B X 30 X 2
        sigma: B X 30 X 2 X 2
        '''
        # Detect dynamic batch size
        batch_size = static.size(0)
        # Detect device
        device = h_0.device

        # Initialize
        x = []
        mu = []
        sigma = []
        h = h_0.unsqueeze(0)
        dx = init_velocity
        x_prev = init_position
        x_flat = torch.zeros_like(z)
        for i in range(30):
          # Unroll a step
          dynamic_encoding, h = self.gru(x_flat.unsqueeze(0), h)
          dynamic_encoding = dynamic_encoding[-1] # Need the last one

          # Concat the dynamic and static encodings
          dynamic_static = torch.cat((dynamic_encoding, static), dim=1)
          # 2-layer MLP
          output = self.mlp(dynamic_static)
          mu_hat = output[:, :2] # [B X 2]
          sigma_hat = output[:, 2:].reshape((batch_size, 2, 2)) # [B X 2 X 2]

          # verlet integration
          mu_ = x_prev + dx + mu_hat
          mu.append(mu_)

          # matrix exponential
          sigma_sym = sigma_hat + sigma_hat.transpose(-2, -1) # Make a symmetric
          
          # "Batched symeig and qr are very slow on GPU"
          # https://github.com/pytorch/pytorch/issues/22573
          sigma_sym = sigma_sym.cpu() # eig decomposition is faster in CPU
          e, v = torch.symeig(sigma_sym, eigenvectors=True)

          # Convert back to gpu tensors
          e = e.to(device)
          v = v.to(device)

          vt = v.transpose(-2, -1)
          e = e.unsqueeze(1) # e: [B X 1 X 2], v: [B X 2 X 2]

          sigma_ = torch.matmul(v * torch.exp(e), vt)
          sigma.append(sigma_)

          # Another way enforcing p.d of sigma
          # sigma_ = torch.matmul(sigma_hat, sigma_hat.transpose(-2, -1)) + torch.eye(2, device=device)
          # sigma.append(sigma_)

          # Transform z to x
          x_ = torch.matmul(sigma_, z[:, 2*i:2*(i+1)].unsqueeze(2)).squeeze(2) + mu_
          x.append(x_)
          dx = x_ - x_prev
          x_prev = x_
          
          # Flattend previous states as next input
          x_flat = torch.zeros_like(z)
          x_flat[:, :2*(i+1)] = torch.cat(x, dim=1)

        # Stack mu's and sigma's as tensors
        mu = torch.stack(mu, dim=1) # mu:  B X 30 X 2
        sigma = torch.stack(sigma, dim=1) # sigma: B X 30 X 2 X 2
        x = torch.stack(x, dim=1) # x: B X 30 X 2

        return x, mu, sigma