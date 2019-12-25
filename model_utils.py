import torch
import torch.nn as nn

import pdb

class R2P2_CNN(nn.Module):
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
    x: B X 6 X 50 X 50
    layer_outputs: (
        B X 32 X 62 X 62,
        B X 32 X 60 X 60,
        B X 32 X 58 X 58,
        B X 32 X 56 X 56,
        B X 32 X 56 X 56,
        B X 32 X 56 X 56,
        B X 32 X 56 X 56,
        B X 32 X 56 X 56,
        B X 32 X 54 X 54,
        B X 32 X 52 X 52,
        B X 32 X 50 X 50
      )
    '''
    conv_modules = self.conv_modules
    activations = self.activations
    layer_outputs = []

    # Conv 1~10 with softplus
    for i in range(1, 11):
      x = conv_modules['conv{:d}'.format(i)](x)
      x = activations['softplus'](x)
      layer_outputs.append(x)

    # Conv 11~12 with tanh
    for i in range(11, 13):
      x = conv_modules['conv{:d}'.format(i)](x)
      x = activations['tanh'](x)
      layer_outputs.append(x)
    
    # Conv 13 (Linear 1X1)
    x = conv_modules['conv13'](x)
    layer_outputs.append(x)

    return x, layer_outputs

class R2P2_Static(nn.Module):
  def __init__(self):
    super(R2P2_Static, self).__init__()
    self.gru = nn.GRU(input_size=2, hidden_size=150, num_layers=1)
    self.mlp = nn.Sequential(
        nn.Linear(150 + 50 * 50 * 6, 50),
        nn.Softplus(),
        nn.Linear(50, 50),
        nn.Softplus()
      )
  
  def forward(self, x, scene):
    '''
    input shape
    x: 20 X B X 2 
    scene: B X 6 X 50 X 50
    
    ouput shape
    static: B X 50
    layer_outputs: (
        B X 150,
        B X 50
      )
    '''
    layer_outputs = []

    motion_encoding, h = self.gru(x)
    motion_encoding = motion_encoding[-1] # Need the last one
    layer_outputs.append(motion_encoding)

    # Flatten the scene encdoing then
    # concat with the motion encoding
    scene_vec = scene.reshape((-1, 6*50*50))
    motion_scene = torch.cat((motion_encoding, scene_vec), dim=1)

    # 2-layer MLP
    static = self.mlp(motion_scene)
    layer_outputs.append(static)

    return static, layer_outputs

class R2P2_Dynamic(nn.Module):
  def __init__(self):
    super(R2P2_Dynamic, self).__init__()
    self.gru = nn.GRU(input_size=30*2, hidden_size=150, num_layers=1)
    self.mlp = nn.Sequential(
        nn.Linear(150+50, 50),
        nn.Softplus(),
        nn.Linear(50, 6),
        nn.Softplus()
    )
  
  @property
  def device(self):
    return next(self.parameters()).device

  def forward(self, static, init_velocity, init_position):
    '''
    input shape
    static: B X 50
    
    ouput shape
    mu: 30 X B X 2
    sigma: 30 X B X 2 X 2
    x : 30 x B X 2
    '''
    # Detect dynamic batch size
    batch_size = static.size(0)

    # Generate latent state z
    z = torch.normal(mean=0.0, std=1.0, size=(30, batch_size, 2), device=self.device)
    # Initial GRU state
    h = torch.zeros((1, batch_size, 150), device=self.device)
    # State Generations
    x = torch.zeros((batch_size, 30, 2), device=self.device)

    mu = []
    sigma = []
    dx = init_velocity
    x_prev = init_position
    for i in range(30):
      # Flattend previous states as input
      x_flat = x.reshape((1, batch_size, 30*2))
      # Unroll a step
      dynamic_encoding, h = self.gru(x_flat, h)
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
      e, v = torch.symeig(sigma_sym, eigenvectors=True)
      vt = v.transpose(-2, -1)
      e = e.unsqueeze(1) # e: [B X 1 X 2], v: [B X 2 X 2]

      sigma_ = torch.matmul(v * torch.exp(e), vt)
      sigma.append(sigma_)

      # Transform z to x
      x_ = torch.matmul(sigma_, z[i].unsqueeze(2)).squeeze(2) + mu_
      x[:, i, :] = x_
      dx = x_ - x_prev
      x_prev = x_
  
    # Stack mu's and sigma's as tensors
    mu = torch.stack(mu) # mu: 30 X B X 2
    sigma = torch.stack(sigma) # sigma: 30 X B X 2 X 2

    # Transpose x
    x = x.transpose(1, 0) # x: 30 X B X 2

    return mu, sigma, x