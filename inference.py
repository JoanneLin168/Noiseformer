import os
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.io as io

from tqdm import tqdm
from models.model import NoiseGenerator, NoiseEstimatorReconNet


class TestArgs:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = '/home/wg19671/data/SIDD_Small_sRGB_Only/Data'               # change to your dataset path
    checkpoint_path = './den.pt'                                                # change to your checkpoint path

args = TestArgs()

# Get all the directories ending in '_L' from root directory]
class SIDDDataset(nn.Module):
    def __init__(self, root, device, subset=None):
        self.root = root
        self.device = device
        self.subset = subset
        self.folders = []
        self.patch_size = 64
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((256, 256)), # change this depending on what it was trained on?
            transforms.CenterCrop(64),
        ])
        print(f"Loading dataset from {self.root}")
        if subset:
            self.folders = [x[0] for x in os.walk(self.root) if x[0].endswith(subset)]
        else:
            self.folders = [x[0] for x in os.walk(self.root)]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        
        video_name = self.folders[idx]

        gt_path = os.path.join(self.root, video_name, "GT_SRGB_010.PNG")
        noisy_path = os.path.join(self.root, video_name, "NOISY_SRGB_010.PNG")

        gt_image = io.read_image(gt_path).to(self.device) / 255.0
        noisy_image = io.read_image(noisy_path).to(self.device) / 255.0
   
        # For an image tensor of shape (C, H, W):
        gt_patches = gt_image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        gt_patches = gt_patches.contiguous().view(gt_image.shape[0], -1, self.patch_size, self.patch_size).permute(1, 0, 2, 3)

        noisy_patches = noisy_image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        noisy_patches = noisy_patches.contiguous().view(noisy_image.shape[0], -1, self.patch_size, self.patch_size).permute(1, 0, 2, 3)

        return noisy_patches, gt_patches
    
test_loader = torch.utils.data.DataLoader(dataset=SIDDDataset(args.dataset_root, args.device, "_L"),
                                        batch_size=1,
                                        shuffle=False)

# Load the model checkpoint (change as appropriate)
model = NoiseGenerator(
    in_channels=3,
    out_channels=3,
    device=args.device,
    patch_size=(64,64)
)

checkpoint = torch.load(
    f = args.checkpoint_path,
    map_location=args.device, 
    weights_only=False
)
model.load_state_dict(checkpoint)
model.to(args.device)
model.eval()

sidd_dir = os.path.join(os.path.dirname(args.checkpoint_path), "sidd_val") 
os.makedirs(sidd_dir, exist_ok=True)

BATCH_SIZE = 16

recorded_labels = []
# for i, (clean, noisy) in enumerate(tqdm(test_loader)):
for noisy in tqdm(sorted(os.listdir("/home/wg19671/data/1709_Outdoor_lowlight_II_NINJAV_S001_S001_T003"))):

    # Load the noisy image
    noisy_path = os.path.join("/home/wg19671/data/1709_Outdoor_lowlight_II_NINJAV_S001_S001_T003", noisy)
    noisy = io.read_image(noisy_path).to(args.device) / 255.0

    # Split into patches
    noisy_patches = noisy.unfold(1, 64, 64).unfold(2, 64, 64)
    noisy_patches = noisy_patches.contiguous().view(noisy.shape[0], -1, 64, 64).permute(1, 0, 2, 3)

    img_labels = []
    for i in tqdm(range(0, noisy_patches.shape[0], BATCH_SIZE), leave = False):
        # Get the current batch of patches
        batch_patches = noisy_patches[i : i + BATCH_SIZE]
                
        synth, recon_imgs, pred_labels = model(batch_patches, batch_patches)
        img_labels.append(pred_labels.mean(dim=0).tolist())

    avg_predicted_label = np.mean(img_labels, axis=0)
    recorded_labels.append(avg_predicted_label)  

# After processing all images, save the list to CSV
df = pd.DataFrame(np.vstack(recorded_labels), columns=['shot_noise', 'read_noise', 'uniform_noise', 'row_noise', 'row_noise_temp', 'periodic0', 'periodic1', 'periodic2'])
csv_path = os.path.join(os.path.dirname(args.checkpoint_path), "predicted_labels.csv")
df.to_csv(csv_path, index=False)
