import os
import json

import torch
import numpy as np
import torchvision.io as io
import copy
from torchvision import transforms

from dataset.noise import StarlightNoise

# For reproducibility
torch.manual_seed(0)

def get_transforms(patch_size=256, split='train'):
    if split == 'train':
        return transforms.Compose([transforms.RandomCrop(patch_size, pad_if_needed=True)])
    else:
        return transforms.Compose([transforms.CenterCrop(patch_size)])

def degradation(image, gt_label=None, device='cuda'):
        if gt_label is None:
            gt_label = {
                'alpha_brightness': torch.tensor(np.round(np.random.uniform(0.05, 0.3), decimals=2)).to(device),
                'gamma_brightness': torch.tensor(np.round(np.random.uniform(0.1, 1), decimals=2)).to(device),
                'shot_noise': torch.rand(1).to(device),
                'read_noise': torch.rand(1).to(device),
                'uniform_noise': torch.rand(1).to(device),
                'row_noise': torch.rand(1).to(device),
                'row_noise_temp': torch.rand(1).to(device),
                'periodic0': torch.rand(1).to(device),
                'periodic1': torch.rand(1).to(device),
                'periodic2': torch.rand(1).to(device),
            }
        
        # Reduce brightness
        alpha = gt_label['alpha_brightness']
        gamma = gt_label['gamma_brightness']
        image = alpha*(torch.pow(image, 1/gamma))
        
        # Passing deep copy to avoid modifying original gt_label
        noisy_image = StarlightNoise(image, copy.deepcopy(gt_label), device=device)
        return noisy_image, gt_label

class UnalignedDegradationDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):

        self.device = args.device
        self.dataset_root = args.dataset_root+f'/{split}_all_frames/JPEGImages'
        self.video_names = sorted(os.listdir(self.dataset_root))
        self.num_vid_frames = args.num_vid_frames

        self.video_dict = {}
        self.frame_dict = {}

        for video in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.dataset_root, video)))
            video_len = len(frame_list)
            self.video_dict[video] = video_len
            self.frame_dict[video] = frame_list

        self.transform = get_transforms(args.patch_size, split)


    def __len__(self):
        return len(self.video_names)
    
    def _sample_index(self, video_length, num_frames):
        if num_frames > video_length:
            print(f"num_frames: {num_frames}, video_length: {video_length}")
            raise ValueError("num_frames should be less than or equal to video_length")
        return torch.randint(0, video_length - num_frames + 1, (1,)).item() + torch.arange(num_frames)

    def __getitem__(self, index):
    
        video_name = self.video_names[index]

        selected_index = self._sample_index(self.video_dict[video_name], self.num_vid_frames)

        frame_list = self.frame_dict[video_name]
        images = []
        noisy_images = []
        gt_labels = []
        gt_label = None

        for idx in selected_index:
            img_path = os.path.join(self.dataset_root, video_name, frame_list[idx])
            image = io.read_image(img_path).to(self.device) / 255.0
            
            # NOTE: must be a tensor already
            image = self.transform(image)

            # Apply noise
            noisy_image, gt_label = degradation(image, gt_label=gt_label)

            # Reduce brightness
            alpha = gt_label['alpha_brightness']
            gamma = gt_label['gamma_brightness']
            dark_image = alpha*(torch.pow(image, 1/gamma))

            images.append(dark_image)
            noisy_images.append(noisy_image)
            
            gt_noise_vals = [gt_label[k].item() for k in gt_label if k != 'alpha_brightness' and k != 'gamma_brightness']
            gt_labels.append(gt_noise_vals)

        images = torch.stack(images, dim=1)
        noisy_images = torch.stack(noisy_images, dim=1)
        gt_labels = torch.tensor(gt_labels)
        noise_types = list(gt_label.keys())

        return {'clean': images.to(self.device),
                'noisy': noisy_images.to(self.device),
                'gt_labels': gt_labels.to(self.device),
                'noise_types': noise_types}


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):

        self.device = args.device
        self.dataset_root = args.dataset_root+f'/{split}_all_frames/JPEGImages'
        self.video_names = sorted(os.listdir(self.dataset_root))
        self.transform = get_transforms(args.patch_size, split)


    def __len__(self):
        return len(self.video_names)
    

    def __getitem__(self, index):
    
        video_name = self.video_names[index]

        frame_root = os.path.join(self.dataset_root, video_name, 'frames')
        noisy_frame_root = os.path.join(self.dataset_root, video_name, 'noisy_frames')
        labels_root = os.path.join(self.dataset_root, video_name, 'labels.json')

        clean_frame_files = sorted(os.listdir(frame_root))
        noisy_frame_files = sorted(os.listdir(noisy_frame_root))

        assert len(clean_frame_files) == len(noisy_frame_files)
            
        images = []
        noisy_images = []

        for file in clean_frame_files:
            img_path = os.path.join(frame_root, file)
            img = io.read_image(img_path).to(self.device) / 255.0
            img = self.transform(img)
            images.append(img)

        for file in noisy_frame_files:
            img_path = os.path.join(noisy_frame_root, file)
            img = io.read_image(img_path).to(self.device) / 255.0
            img = self.transform(img)
            noisy_images.append(img)

        images = torch.stack(images, dim=1)
        noisy_images = torch.stack(noisy_images, dim=1)

        with open(labels_root, 'r') as f:
            labels = json.load(f)
        # Assume the labels json is a dictionary
        noise_types = list(labels.keys())
        gt_labels = torch.tensor(list(labels.values()))

        return {'clean': images.to(self.device),
                'noisy': noisy_images.to(self.device),
                'gt_labels': gt_labels.to(self.device),
                'noise_types': noise_types}
