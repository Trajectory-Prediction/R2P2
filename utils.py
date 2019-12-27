import time

import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
            valid_ade, valid_fde, scheduler_metric = self.inference()
            self.scheduler.step(scheduler_metric)

            print_out("------------------------------------------------------------------------------------------", self.text_logger)
            # if self.discriminator is None:
            #     print_out(f'| Epoch: {epoch:02} | Train Loss: {train_loss:0.6f} | Train ADE: {train_ade:0.4f} | Train FDE: {train_fde:0.4f}', self.text_logger)
            # else:
            #     print_out(f'| Epoch: {epoch:02} | Train G_Loss: {train_g_loss:0.6f} | Train D_Loss: {train_d_loss:0.6f} |Train ADE: {train_ade:0.4f} | Train FDE: {train_fde:0.4f}', self.text_logger)

            # print_out(f'| Epoch: {epoch:02} | Valid ADE: {valid_ade:0.4f} | Valid FDE: {valid_fde:0.4f} | Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_lr():g}\n', self.text_logger)

            self.save_checkpoint(epoch, ade=valid_ade, fde=valid_fde)

            # # Log values to Tensorboard
            # if self.discriminator is None:
            #     self.logger.add_scalar('data/Train Loss', train_loss, epoch)
            #     self.logger.add_scalar('data/Learning Rate', self.get_lr(), epoch)
            # else:
            #     self.logger.add_scalar('data/Train G_Loss', train_g_loss, epoch)
            #     self.logger.add_scalar('data/Train D_Loss', train_d_loss, epoch)
            #     self.logger.add_scalar('data/G_Learning Rate', self.get_lr(), epoch)
            #     self.logger.add_scalar('data/D_Learning Rate', self.get_D_lr(), epoch)

            # self.logger.add_scalar('data/Train ADE', train_ade, epoch)
            # self.logger.add_scalar('data/Train FDE', train_fde, epoch)
            # self.logger.add_scalar('data/Scheduler Metric', scheduler_metric, epoch)

            # self.logger.add_scalar('data/Valid ADE', valid_ade, epoch)
            # self.logger.add_scalar('data/Valid FDE', valid_fde, epoch)

        self.logger.close()
        print_out("Training Complete! ", self.text_logger)

    def train_single_epoch(self):
        """Trains the model for a single round."""

        self.model.train()
        epoch_loss = 0.0
        epoch_ade, epoch_fde = 0.0, 0.0
        epoch_agents = 0.0
        for b, batch in enumerate(self.train_loader):
            print("Working on batch {:d}/{:d}".format(b+1, len(self.train_loader)), end='\r')        
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
            z = torch.normal(mean=0.0, std=1.0, size=(30, batch_size, 2)).to(self.device)
            # Initial GRU state
            h_0 = torch.zeros((1, batch_size, 150)).to(self.device)

            gen_trajs, mu, sigma  = self.model(z, h_0, src_trajs, src_lens, decode_start_vel, decode_start_pos, scene_images)

            # Prior Loss (p loss)
            ploss = self.criterion(gen_trajs, tgt_trajs)
            ploss = ploss.mean()

            # Normalizing Flow (q loss)
            gen_trajs_bt = gen_trajs.reshape((-1, 2))
            mu_bt = mu.reshape((-1, 2))

            traj_mu = (gen_trajs_bt - mu_bt).unsqueeze(2)
            sigma_bt = sigma.reshape((-1, 2, 2))

            z_, _ = torch.solve(traj_mu, sigma_bt)

            # TODO
            # batch_loss = q_loss + p_loss

            # with torch.no_grad():
            #     error = predicted_trajs - tgt_trajs
            #     sq_error = (error ** 2).sum(2).sqrt()
            #     sq_error = sq_error.reshape((-1))

            #     # ADE
            #     batch_ade = batch_ade.mean()

            #     # FDE
            #     batch_fde = batch_fde.mean()

            # batch_loss.backward()
            # self.optimizer.step()

            # epoch_loss += batch_loss.item()
            # epoch_ade += batch_ade.item()
            # epoch_fde += batch_fde.item()
            # epoch_agents += len(tgt_lens)

        # epoch_loss /= (b+1)
        # epoch_ade /= epoch_agents
        # epoch_fde /= epoch_agents
        return epoch_loss, epoch_ade, epoch_fde

    # TODO
    # Implement visualization in inference function
    # Write inference code based on the train_single_eopch
    # Be advised that call self.model.eval() before inference
    # And wrap the inference codes under with torch.no_grad()
    # Otherwise memory will explode.

    # def inference(self):
    #     self.model.eval()  # Set model to evaluate mode.
        
    #     with torch.no_grad():
    #         epoch_ade, epoch_fde = 0.0, 0.0
    #         epoch_agents = 0.0

    #         for b, batch in enumerate(self.valid_loader):
    #             scene_images, agent_masks, num_src_trajs, src_trajs, src_lens, unsorter, num_tgt_trajs, tgt_trajs, tgt_lens, encode_coords, decode_rel_pos, decode_start_pos = batch
    #             scene_images = scene_images.to(self.device, non_blocking=True)
    #             src_trajs = src_trajs.to(self.device, non_blocking=True)
    #             src_lens = src_lens.to(self.device, non_blocking=True)
    #             tgt_trajs = tgt_trajs.to(self.device, non_blocking=True)
    #             decode_rel_pos = decode_rel_pos.to(self.device, non_blocking=True)
    #             decode_start_pos = decode_start_pos.to(self.device, non_blocking=True)

    #             # Prediction
    #             predicted_trajs = self.model(src_trajs, src_lens, unsorter, agent_masks,
    #                                         decode_rel_pos[agent_masks], decode_start_pos[agent_masks],
    #                                         self.stochastic, encode_coords, scene_images)

    #             # Calculate the sample indices
    #             time_normalizer = []
    #             all_agent_time_index = []
    #             all_agent_final_time_index = []

    #             for i, tgt_len in enumerate(tgt_lens):
    #                 idx_i = np.arange(tgt_len) + i*30
    #                 normalizer_i = torch.ones(tgt_len) * tgt_len
    #                 time_normalizer.append(normalizer_i)
    #                 all_agent_time_index.append(idx_i)
    #                 all_agent_final_time_index.append(idx_i[-1])

    #             time_normalizer = torch.cat(time_normalizer).to(self.device)
    #             all_agent_time_index = np.concatenate(all_agent_time_index)

    #             error = predicted_trajs - tgt_trajs
    #             sq_error = (error ** 2).sum(2).sqrt()
    #             sq_error = sq_error.reshape((-1))

    #             # ADE
    #             batch_ade = sq_error[all_agent_time_index]
    #             batch_ade /= time_normalizer
    #             batch_ade = batch_ade.sum()

    #             # FDE
    #             batch_fde = sq_error[all_agent_final_time_index]
    #             batch_fde = batch_fde.sum()

    #             epoch_ade += batch_ade.item()
    #             epoch_fde += batch_fde.item()
    #             epoch_agents += len(tgt_lens)

    #         epoch_ade /= epoch_agents
    #         epoch_fde /= epoch_agents

    #     scheduler_metric = (epoch_ade + epoch_fde) / 2.0

    #     return epoch_ade, epoch_fde, scheduler_metric

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
