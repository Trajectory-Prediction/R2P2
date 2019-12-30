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
            nn.Linear(50, 6),
            nn.Softplus()
        )

    def forward(self, z, h_0, static, init_velocity, init_position):
        '''
        input shape
        z: 30 X B X 2
        h_0: 1 X B X 150
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

        # State Generations
        x = torch.zeros((batch_size, 30, 2), device=device)

        mu = []
        sigma = []
        h = h_0
        dx = init_velocity
        x_prev = init_position

        grutime = 0.0
        mlptime = 0.0
        exptime = 0.0
        transformtime = 0.0

        for i in range(30):
          # Flattend previous states as input
          x_flat = x.reshape((1, batch_size, 30*2))

          start_time = time.time()

          # Unroll a step
          dynamic_encoding, h = self.gru(x_flat, h)
          dynamic_encoding = dynamic_encoding[-1] # Need the last one

          grutime += time.time() - start_time

          start_time = time.time()

          # Concat the dynamic and static encodings
          dynamic_static = torch.cat((dynamic_encoding, static), dim=1)
          # 2-layer MLP
          output = self.mlp(dynamic_static)
          mu_hat = output[:, :2] # [B X 2]
          sigma_hat = output[:, 2:].reshape((batch_size, 2, 2)) # [B X 2 X 2]

          mlptime += time.time() - start_time

          # verlet integration
          mu_ = x_prev + dx + mu_hat
          mu.append(mu_)

          

          # matrix exponential
          # sigma_sym = sigma_hat + sigma_hat.transpose(-2, -1) # Make a symmetric
          
          # start_time = time.time()
          # pdb.set_trace()
          # e, v = torch.symeig(sigma_sym, eigenvectors=True)

          # exptime += time.time() - start_time

          # vt = v.transpose(-2, -1)
          # e = e.unsqueeze(1) # e: [B X 1 X 2], v: [B X 2 X 2]

          # sigma_ = torch.matmul(v * torch.exp(e), vt)
          # sigma.append(sigma_)

          # p.d. sigma enforcement
          sigma_ = torch.matmul(sigma_hat, sigma_hat.transpose(-2, -1)) + 1e-5 * torch.eye(2, device=device)
          sigma.append(sigma_)

          start_time = time.time()

          # Transform z to x
          x_ = torch.matmul(sigma_, z[i].unsqueeze(2)).squeeze(2) + mu_
          x = x.clone()
          x[:, i, :] = x_
          dx = x_ - x_prev
          x_prev = x_
      
          transformtime = time.time() - start_time
        
        # print(grutime, mlptime, exptime, transformtime)

        # Stack mu's and sigma's as tensors
        mu = torch.stack(mu, dim=1) # mu:  B X 30 X 2
        sigma = torch.stack(sigma, dim=1) # sigma: B X 30 X 2 X 2

        return x, mu, sigma