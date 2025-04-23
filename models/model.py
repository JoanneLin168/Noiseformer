import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import Unet
from dataset.noise import StarlightNoise

class NoiseEstimatorReconNet(nn.Module):
    """
    This will be the very basic CNN model we will use for the regression task.
    """
    def __init__(self, in_channels=1, noise_list='shot_read_uniform_row1_rowt_periodic',
                 opts='residualFalse_conv_tconv_selu', device='cuda',patch_size=(256, 256)):
        super(NoiseEstimatorReconNet, self).__init__()
        self.num_classes = len(noise_list.split('_'))
        if 'periodic' in noise_list:
            self.num_classes += 2 # for periodic1 and periodic2
        self.in_channels = in_channels
        self.noise_list = noise_list
        self.device = device

        res_opt = bool(opts.split('_')[0].split('residual')[-1]) 
        self.model = Unet(n_channel_in=in_channels, 
                            n_channel_out=in_channels, 
                            residual=res_opt, 
                            down=opts.split('_')[1], 
                            up=opts.split('_')[2], 
                            activation=opts.split('_')[3])
        self.embed_dim = 256

        self.initialized = False
        self.conv1 = nn.Conv2d(in_channels=self.embed_dim, out_channels=2*self.embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2*self.embed_dim, out_channels=2*self.embed_dim, kernel_size=3, stride=1, padding=1)

        div = 2**4 # 4 downsample layers
        self.linear_line_size = int(2*self.embed_dim * (patch_size[-2]//div)*(patch_size[-1]//div))
        self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)
        self.act = nn.Tanh()
        
    def forward(self, x):
        c0, c1, c2, c3, c4, x_enc = self.model(x, task="encode")

        x = self.conv1(x_enc)
        x = nn.functional.relu(x)
        # x = self.pool1(x)
        x = self.conv2(x)
        x = self.act(x)
        # x = self.pool2(x)
 
        x = x.view(-1, self.linear_line_size)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        x_recon = self.model((c0, c1, c2, c3, c4, x_enc), task="decode")

        return x, x_recon # idx 0=shot_noise, 1=read_noise, 2=uniform_noise, 3=row_noise, 4=row_noise_temp, 5=periodic0, 6=periodic1, 7=periodic2
  

class NoiseGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, patch_size=(256,256),
                 noise_list='shot_read_uniform_row1_rowt_periodic', opts='residualFalse_conv_tconv_selu', device='cuda'):
        super(NoiseGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.noise_list = noise_list
        self.device = device
        self.estimator = NoiseEstimatorReconNet(in_channels=3, noise_list=noise_list, patch_size=patch_size).to(device)


    def forward(self, clean, noisy):
        """
        Pass in noisy image, and clean image
        """

        noise_params, x_recon = self.estimator(noisy)

        noise_dict = {}
        bs = clean.shape[0]
        for i, noise in enumerate(self.noise_list.split('_')):
            key = noise+'_noise'
            if noise == 'periodic':
                noise_dict['periodic0'] = noise_params[:,i].view(bs, 1)
                noise_dict['periodic1'] = noise_params[:,i+1].view(bs, 1)
                noise_dict['periodic2'] = noise_params[:,i+2].view(bs, 1)
                i += 2
            elif noise == 'row1':
                noise_dict['row_noise'] = noise_params[:,i].view(bs, 1, 1, 1)
            elif noise == 'rowt':
                noise_dict['row_noise_temp'] = noise_params[:,i].view(bs, 1, 1, 1)
            else:
                noise_dict[key] = noise_params[:,i].view(bs, 1, 1, 1)


        x = StarlightNoise(clean, noise_dict)

        return x, x_recon, noise_params
