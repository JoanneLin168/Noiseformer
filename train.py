import sys, os
import numpy as np
import torch
from datetime import datetime
import argparse, json
import random

from torch.utils.tensorboard import SummaryWriter

from dataset.degradation_dataset import UnalignedDegradationDataset, ValidationDataset
from models.model import NoiseGenerator
from src.trainer import Trainer

os.environ["USE_LIBUX"] = "0"

def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_arguments():
    parser = argparse.ArgumentParser(description='Gan noise model training options.')
    parser.add_argument('--dataset_root', default="./data", help='Path to dataset')
    parser.add_argument('--output_folder', default = './runs/', help='Specify where to save checkpoints during training')

    parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint to load')

    parser.add_argument('--num_vid_frames', default=16, type=int, help='Number of frames to sample from each video')
    parser.add_argument('--noiselist', default='shot_read_uniform_row1_rowt_periodic', help = 'Specify the type of noise to include. \Options: read, shot, uniform, row1, rowt, periodic')
    parser.add_argument('--unet_opts', default='residualFalse_conv_tconv_selu')
    parser.add_argument('--w1', default=1, type=float, help='Weight for MLP loss')
    parser.add_argument('--w2', default=1, type=float, help='Weight for recon loss')
    
    parser.add_argument('--lr', default = 0.0002, type=float)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--batch_size', default = 2, type=int)
    parser.add_argument('--patch_size', default=64, type=int) 
    parser.add_argument('--eval_patch_size', default=64, type=int)
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--display_freq', default=500, type=int, help='Frequency of visualizing training results')
    parser.add_argument('--log_freq', default=100, type=int, help='Frequency of logging training results')
    parser.add_argument('--save_freq', default=1, type=int, help='Frequency of saving checkpoints')

    parser.add_argument('--device', default= 'cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
    parser.add_argument('--notes', default='.', type=str, help='Add notes to the experiment')

    args = parser.parse_args()

    return args
    
def main(args):

    # Misc setup
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_folder = args.output_folder + 'run_' + start_time + '/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(output_folder + 'checkpoints')

    with open(output_folder + 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    args.output_folder = output_folder
    writer = SummaryWriter(args.output_folder)
        
    # Training setup
    train_set = UnalignedDegradationDataset(args, split='train')
    val_set = ValidationDataset(args, split='valid')
    model = NoiseGenerator(in_channels=3, out_channels=3, noise_list=args.noiselist, patch_size=(args.patch_size, args.patch_size), device=args.device)
    model.to(args.device)

    # Create trainer
    trainer = Trainer(args, model, train_set, val_set, writer)
    trainer.run()

    writer.close()
    print("Finished training!")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')

    args = get_arguments()
    set_seed(args.seed)
    main(args)
