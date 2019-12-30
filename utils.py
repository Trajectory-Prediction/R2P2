import time

import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

import pdb

# EPS = 1e-9

# def bilinear_interpolation_1d(prediction, prior):
    
#     # Change to map CS
#     prediction_mapCS = prediction * 2.0 + 112.0
#     prediction_mapCS = prediction_mapCS.transpose(0, 1)

#     x = prediction_mapCS[:, :, 0].reshape(-1)
#     y = prediction_mapCS[:, :, 1].reshape(-1)

#     # Filter out-of-range coordinates
#     x_out_left = x < 0
#     x_out_right = x > 223
#     y_out_left = y < 0
#     y_out_right = y > 223

#     out_mask = x_out_left | x_out_right | y_out_left | y_out_right
#     in_mask = torch.logical_not(out_mask)

#     x[out_mask] = 0.0
#     y[out_mask] = 0.0

#     # Detect batch_size
#     batch_size = prior.size(0)
#     batch_mask = []
#     for i in range(batch_size):
#         batch_mask.extend([i] * 30)
#     batch_mask = torch.LongTensor(batch_mask)

#     # Qunatize x and y
#     x1 = torch.floor(x)
#     x2 = torch.ceil(x)
#     y1 = torch.floor(y)
#     y2 = torch.ceil(y)

#     x1_int = x1.long()
#     x2_int = x2.long()
#     y1_int = y1.long()
#     y2_int = y2.long()

#     q11 = prior[batch_mask, y1_int, x1_int]
#     q12 = prior[batch_mask, y1_int, x2_int]
#     q21 = prior[batch_mask, y2_int, x1_int]
#     q22 = prior[batch_mask, y2_int, x2_int]
    
#     result = (q11 * ((x2 - x) * (y2 - y)) +
#               q21 * ((x - x1) * (y2 - y)) +
#               q12 * ((x2 - x) * (y - y1)) +
#               q22 * ((x - x1) * (y - y1))
#             )
    
#     # Assign 0 prob. to out masks
#     result[out_mask] = EPS
#     result[result < EPS] = EPS

#     return result

def ADE(predicted_traj, future_traj):
    # predicted_traj : [B X T X 2]
    # future_traj : [B X T X 2]
    # Average over the time
    error = predicted_traj - future_traj
    return (error**2).sum(2).sqrt().mean(1)

def FDE(predicted_traj, future_traj):
    # predicted_traj : [B X T X 2]
    # future_traj : [B X T X 2]
    # Only the last time
    error = predicted_traj[:, -1, :] - future_traj[:, -1, :]
    return (error**2).sum(1).sqrt()

def print_out(string, file_front):
    print(string)
    print(string, file=file_front)

class ModelTrainer:
    def __init__(self, model, train_loader, valid_loader, criterion,
                 optimizer, exp_path, text_logger, logger,
                 device, load_ckpt=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=(1/2), verbose=True, patience=3)
        self.exp_path = exp_path
        self.text_logger = text_logger
        self.logger = logger
        self.device = device
        self.beta = 1.0

        if load_ckpt:
            self.load_checkpoint(load_ckpt)

        # Other Parameters
        self.best_valid_ade = None
        self.best_valid_fde = None
        self.start_epoch = 1

    def train(self, num_epochs):
        print_out('TRAINING .....', self.text_logger)
        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            print_out("==========================================================================================", self.text_logger)

            train_loss, train_ade, train_fde = self.train_single_epoch()
            valid_loss, valid_ade, valid_fde, scheduler_metric = self.inference()
            self.scheduler.step(scheduler_metric)

            print_out("------------------------------------------------------------------------------------------", self.text_logger)
            # if self.discriminator is None:
            print_out(f'| Epoch: {epoch:02} | Train Loss: {train_loss:0.6f} | Train ADE: {train_ade:0.4f} | Train FDE: {train_fde:0.4f}', self.text_logger)
            # else:
            #     print_out(f'| Epoch: {epoch:02} | Train G_Loss: {train_g_loss:0.6f} | Train D_Loss: {train_d_loss:0.6f} |Train ADE: {train_ade:0.4f} | Train FDE: {train_fde:0.4f}', self.text_logger)

            print_out(f'| Epoch: {epoch:02} | Valid Loss: {valid_loss:0.6f} | Valid ADE: {valid_ade:0.4f} | Valid FDE: {valid_fde:0.4f} | Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_lr():g}\n', self.text_logger)

            self.save_checkpoint(epoch, ade=valid_ade, fde=valid_fde)

            # Log values to Tensorboard
            # if self.discriminator is None:
            self.logger.add_scalar('data/Train Loss', train_loss, epoch)
            self.logger.add_scalar('data/Learning Rate', self.get_lr(), epoch)
            # else:
            #     self.logger.add_scalar('data/Train G_Loss', train_g_loss, epoch)
            #     self.logger.add_scalar('data/Train D_Loss', train_d_loss, epoch)
            #     self.logger.add_scalar('data/G_Learning Rate', self.get_lr(), epoch)
            #     self.logger.add_scalar('data/D_Learning Rate', self.get_D_lr(), epoch)

            self.logger.add_scalar('data/Train ADE', train_ade, epoch)
            self.logger.add_scalar('data/Train FDE', train_fde, epoch)
            self.logger.add_scalar('data/Scheduler Metric', scheduler_metric, epoch)

            self.logger.add_scalar('data/Valid Loss', valid_loss, epoch)
            self.logger.add_scalar('data/Valid ADE', valid_ade, epoch)
            self.logger.add_scalar('data/Valid FDE', valid_fde, epoch)

        self.logger.close()
        print_out("Training Complete! ", self.text_logger)

    def train_single_epoch(self):
        """Trains the model for a single round."""
        
        self.model.train()
        epoch_loss = 0.0
        epoch_ade, epoch_fde = 0.0, 0.0
        epoch_agents = 0.0

        for b, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            src_trajs, src_lens, tgt_trajs, tgt_lens, decode_start_vel, decode_start_pos, decode_start_pos_city, scene_images, scene_id = batch

            # Detect dynamic batch size
            batch_size = scene_images.size(0)

            scene_images = scene_images.to(self.device, non_blocking=True)
            src_trajs = src_trajs.to(self.device, non_blocking=True)
            src_lens = src_lens.to(self.device, non_blocking=True)
            tgt_trajs = tgt_trajs.to(self.device, non_blocking=True)
            decode_start_vel = decode_start_vel.to(self.device, non_blocking=True)
            decode_start_pos = decode_start_pos.to(self.device, non_blocking=True)

            # Generate latent state z
            z = torch.normal(mean=0.0, std=1.0, size=(batch_size, 30*2)).to(self.device)
            # Initial GRU state
            h_0 = torch.zeros((batch_size, 150)).to(self.device)

            gen_trajs, mu, sigma  = self.model(z, h_0, src_trajs, src_lens, decode_start_vel, decode_start_pos, scene_images)

            # Prior Loss (p loss)
            ploss = self.criterion(gen_trajs, tgt_trajs)
            ploss = ploss.sum(dim=2) # [32 X 30]
            ploss = ploss.mean(dim=1) # [32]

            # Normalizing Flow (q loss)
            tgt_trajs_bt = tgt_trajs.reshape((-1, 2)) # 960 X 2
            mu_bt = mu.reshape((-1, 2)) # 960 X 2

            perterb_traj_bt = torch.normal(mean=0.0, std=0.001, size=tgt_trajs_bt.shape).to(self.device)

            traj_mu = (tgt_trajs_bt - mu_bt - perterb_traj_bt).unsqueeze(2) # 960 X 2 X 1
            sigma_bt = sigma.reshape((-1, 2, 2)) # 960 X 2 X 2

            z_, _ = torch.solve(traj_mu, sigma_bt) # [960 X 2 X 1]
            z_ = z_.reshape((batch_size, 1, -1)) # [32 X 1 X 60]

            c1 = -55.13631199228036 # -30 * log(2 * pi)
            log_q0 = c1 - 0.5 * (torch.matmul(z_, z_.transpose(-1, -2)))
            log_q0 = log_q0.squeeze() # [32]

            det_sigma = torch.det(sigma_bt).reshape((batch_size, -1)) # [32 * 30]

            log_qpi = log_q0 - torch.log(det_sigma).sum(dim=1) # [32]
            

            batch_loss = (-log_qpi + self.beta * ploss).mean()
            batch_loss.backward()

            self.optimizer.step()

            # if b == 100:
            #     pdb.set_trace()

            # print("datatime: {:.2f}, gentime: {:.2f}, nftime: {:.2f}, bptime: {:.2f}".format(datatime, gentime, nftime, bptime))
            with torch.no_grad():
                # Loss
                batch_qloss = -log_qpi.mean().item()
                batch_ploss = ploss.mean().item()

                # ADE
                batch_ade = ADE(gen_trajs, tgt_trajs)
                batch_ade = batch_ade.sum()

                # FDE
                batch_fde = FDE(gen_trajs, tgt_trajs)
                batch_fde = batch_fde.sum()

            print("Working on train batch {:d}/{:d}... batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:.2f}, ade: {:.2f}, fde: {:.2f}".format(b+1, len(self.train_loader), batch_loss.item(), batch_qloss, batch_ploss, batch_ade.item() / batch_size, batch_fde.item() / batch_size), end='\r')

            epoch_loss += batch_loss.item()
            epoch_ade += batch_ade.item()
            epoch_fde += batch_fde.item()
            epoch_agents += len(tgt_lens)

        epoch_loss /= (b+1)
        epoch_ade /= epoch_agents
        epoch_fde /= epoch_agents

        return epoch_loss, epoch_ade, epoch_fde

    # TODO
    # Implement visualization in inference function
    # Write inference code based on the train_single_eopch
    # Be advised that call self.model.eval() before inference
    # And wrap the inference codes under with torch.no_grad()
    # Otherwise memory will explode.

    def inference(self):
        self.model.eval()  # Set model to evaluate mode.
        
        epoch_ade, epoch_fde = 0.0, 0.0
        epoch_loss = 0.0
        epoch_agents = 0.0

        with torch.no_grad():
            for b, batch in enumerate(self.valid_loader):
                print("Working on validation batch {:d}/{:d}".format(b+1, len(self.train_loader)), end='\r')

                src_trajs, src_lens, tgt_trajs, tgt_lens, decode_start_vel, decode_start_pos, decode_start_pos_city, scene_images, scene_id = batch
                
                # Detect dynamic batch size
                batch_size = scene_images.size(0)

                scene_images = scene_images.to(self.device, non_blocking=True)
                src_trajs = src_trajs.to(self.device, non_blocking=True)
                src_lens = src_lens.to(self.device, non_blocking=True)
                tgt_trajs = tgt_trajs.to(self.device, non_blocking=True)
                decode_start_vel = decode_start_vel.to(self.device, non_blocking=True)
                decode_start_pos = decode_start_pos.to(self.device, non_blocking=True)

                # Generate latent state z
                z = torch.normal(mean=0.0, std=1.0, size=(batch_size, 30*2)).to(self.device)
                # Initial GRU state
                h_0 = torch.zeros((batch_size, 150)).to(self.device)

                gen_trajs, mu, sigma  = self.model(z, h_0, src_trajs, src_lens, decode_start_vel, decode_start_pos, scene_images)

                # Prior Loss (p loss)
                ploss = self.criterion(gen_trajs, tgt_trajs)
                ploss = ploss.sum(dim=2) # [32 X 30]
                ploss = ploss.mean(dim=1) # [32]

                # Normalizing Flow (q loss)
                tgt_trajs_bt = tgt_trajs.reshape((-1, 2)) # 960 X 2
                mu_bt = mu.reshape((-1, 2)) # 960 X 2

                perterb_traj_bt = torch.normal(mean=0.0, std=0.001, size=tgt_trajs_bt.shape).to(self.device)

                traj_mu = (tgt_trajs_bt - mu_bt - perterb_traj_bt).unsqueeze(2) # 960 X 2 X 1
                sigma_bt = sigma.reshape((-1, 2, 2)) # 960 X 2 X 2

                z_, _ = torch.solve(traj_mu, sigma_bt) # [960 X 2 X 1]
                z_ = z_.reshape((batch_size, 1, -1)) # [32 X 1 X 60]

                c1 = -55.13631199228036 # -30 * log(2 * pi)
                log_q0 = c1 - 0.5 * (torch.matmul(z_, z_.transpose(-1, -2)))
                log_q0 = log_q0.squeeze() # [32]

                det_sigma = torch.det(sigma_bt).reshape((batch_size, -1)) # [32 * 30]

                log_qpi = log_q0 - torch.log(det_sigma).sum(dim=1) # [32]

                batch_loss = (-log_qpi + self.beta * ploss).mean()

                # ADE
                batch_ade = ADE(gen_trajs, tgt_trajs)
                batch_ade = batch_ade.sum()
                # FDE
                batch_fde = FDE(gen_trajs, tgt_trajs)
                batch_fde = batch_fde.sum()

                epoch_loss += batch_loss.item()
                epoch_ade += batch_ade.item()
                epoch_fde += batch_fde.item()
                epoch_agents += len(tgt_lens)

        epoch_loss /= (b+1)
        epoch_ade /= epoch_agents
        epoch_fde /= epoch_agents

        scheduler_metric = (epoch_ade + epoch_fde) / 2.0

        return epoch_loss, epoch_ade, epoch_fde, scheduler_metric

    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_D_lr(self):
        for param_group in self.optimizer_D.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, ade, fde):
        """Saves experiment checkpoint.
        Saved state consits of epoch, model state, optimizer state, current
        learning rate and experiment path.
        """

        state_dict = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.get_lr(),
            'exp_path': self.exp_path,
            'val_ade': ade,
            'val_fde': fde,
        }

        save_path = "{}/ck_{}_{:0.4f}_{:0.4f}.pth.tar".format(self.exp_path, epoch, ade, fde)
        torch.save(state_dict, save_path)

    def load_checkpoint(self, ckpt):
        print_out("Loading checkpoint from {:s}".format(ckpt), self.text_logger)
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)

        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.start_epoch = checkpoint['epoch']
