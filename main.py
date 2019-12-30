import os
import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import R2P2_CNN, R2P2_RNN
from dataset import ArgoverseDataset
from model_utils import ContextEncoder, DynamicDecoder
from utils import ModelTrainer

import pdb

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_path = '/mnt/sdb1/shpark/cmu_mmml_project/r2p2_experiments/' + args.tag + '_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).strftime('_%d_%B__%H_%M_')

    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    print(f"Current Exp Path: {exp_path}")

    logger = SummaryWriter(exp_path + '/logs')

    train_dataset = ArgoverseDataset(args.train_dir, map_version=args.map_version, testset=False,
                                     num_workers=args.num_workers, cache_file=args.train_cache)
    valid_dataset = ArgoverseDataset(args.valid_dir, map_version=args.map_version, testset=False,
                                     num_workers=args.num_workers, cache_file=args.val_cache)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

    print(f'Train Examples: {len(train_dataset)} | Valid Examples: {len(valid_dataset)}')

    __context_encoder = ContextEncoder()
    __dynamic_decoder = DynamicDecoder()

    # Configure model based on the selected option:
    if args.model_type == 'R2P2_RNN':
        model = R2P2_RNN(context_encoder=__context_encoder, dynamic_decoder=__dynamic_decoder)
    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    # Send model to Device:
    model = model.to(device)

    # if args.gpu_devices:
    #     model = nn.DataParallel(model, device_ids=eval(args.gpu_devices))

    if args.criterion == 'mseloss':
        criterion = torch.nn.MSELoss(reduction='none')

    criterion = criterion.to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                     weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=0.9, weight_decay=1e-4)

    output_log = open(exp_path + '/output_log.txt', 'w')

    trainer = ModelTrainer(model, train_loader, valid_loader, criterion, optimizer,
                           exp_path, output_log, logger, device, args.load_ckpt)

    trainer.train(args.num_epochs)
    output_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Tag
    parser.add_argument('--tag', type=str, help="Add a tag to the saved folder")

    # Misc Parameters
    parser.add_argument('--train_dir', type=str,
                        default='/home/spalab/argoverse_shpark/argoverse-forecasting-from-forecasting/train/',
                        help="Train Directory")
    parser.add_argument('--valid_dir', type=str,
                        default='/home/spalab/argoverse_shpark/argoverse-forecasting-from-forecasting/val/',
                        help="Valid Directory")

    parser.add_argument('--train_cache', type=str, help="")
    parser.add_argument('--val_cache', type=str, help="")


    parser.add_argument('--model_type', type=str, default='R2P2_RNN', help="R2P2_RNN")
    parser.add_argument('--map_version', type=str, default='1.3', help="Map version")
    
    
    # Training Parameters
    parser.add_argument('--criterion', type=str, default='mseloss', help="Training Criterion")
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training the model")
    parser.add_argument('--num_workers', type=int, default=24, help="")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('-load', '--load_ckpt', default=None, help='Load Model Checkpoint')

    parser.add_argument('--gpu_devices', type=str, default='1', help="Use Multiple GPUs for training")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    main(args)