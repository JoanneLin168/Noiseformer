import torch
import torch.nn as nn
import numpy as np
import scipy.io


# for sampling, use actual value of: (vmax-vmin)*label + vmin, as label is scaled to be between 0 and 1
# NOTE: for brightness it is already in the right range, as it isn't a value to be predicted
actual_labels = {
    'shot_noise': [0, 0.5],
    'read_noise': [0, 0.1],
    'uniform_noise': [0, 0.1],
    'row_noise': [0, 0.01],
    'row_noise_temp': [0, 0.01],
    'periodic0': [0, 0.5],
    'periodic1': [0, 0.5],
    'periodic2': [0, 0.5],
}

def StarlightNoise(x, noise_dict_not_scaled, keep_track=False, device='cuda'):

    all_noise = {}

    assert x.min() >= 0 and x.max() <= 1, "Input tensor should be in [0, 1] range"

    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True

    # Scale noise dict back to actual values: (vmax-vmin)*label + vmin
    noise_dict = {}
    for key in actual_labels:
        scale = actual_labels[key][1] - actual_labels[key][0]
        noise_dict[key] = scale*noise_dict_not_scaled[key] + actual_labels[key][0]
    
    noise = torch.zeros_like(x)
    # Read and shot noise
    variance = x*noise_dict['shot_noise'] + noise_dict['read_noise']
    shot_noise = torch.randn(x.shape, device=device)*variance
    noise += shot_noise
    if keep_track == True:
        all_noise['shot_read'] = shot_noise.detach().cpu().numpy() 

    # Uniform noise
    uniform_noise = noise_dict['uniform_noise']*torch.rand(x.shape, device=device)
    noise += uniform_noise
    if keep_track == True:
        all_noise['uniform'] = uniform_noise.detach().cpu().numpy() 

    # Row 1 noise
    row_noise = noise_dict['row_noise']*torch.randn([*x.shape[0:-2],x.shape[-1]], device=device).unsqueeze(-2)
    noise += row_noise
    if keep_track == True:
        all_noise['row'] = np.repeat(row_noise.detach().cpu().numpy(), all_noise['shot_read'].shape[-2], axis=-2)

    # Row temp noise  
    row_noise_temp = noise_dict['row_noise_temp']*torch.randn([*x.shape[0:-3],x.shape[-1]], device=device).unsqueeze(-2).unsqueeze(-2)
    noise += row_noise_temp
    if keep_track == True:
        all_noise['row_temp'] = np.repeat(row_noise_temp.detach().cpu().numpy(), all_noise['shot_read'].shape[-2], axis=-2)

    # Periodic noise
    periodic_param0 = noise_dict['periodic0']
    periodic_param1 = noise_dict['periodic1']
    periodic_param2 = noise_dict['periodic2']

    periodic_noise = torch.zeros(x.shape,  dtype=torch.cfloat, device=device)
    periodic_noise[...,0,0] = periodic_param0*torch.randn((x.shape[0:-2]), device=device)
    
    periodic0 = periodic_param1*torch.randn((x.shape[0:-2]), device=device)
    periodic1 = periodic_param2*torch.randn((x.shape[0:-2]), device=device) 

    periodic_noise[...,0,x.shape[-1]//4] = torch.complex(periodic0, periodic1)
    periodic_noise[...,0,3*x.shape[-1]//4] = torch.complex(periodic0, -periodic1)

    periodic_gen = torch.abs(torch.fft.ifft2(periodic_noise, norm="ortho"))

    noise += periodic_gen
    if keep_track == True:
        all_noise['periodic'] = periodic_gen.detach().cpu().numpy() 
    
    # Add noise to image
    noisy = x + noise
    noisy = torch.clip(noisy, 0, 1)

    if squeeze:
        x = x.squeeze(0)
        noisy = noisy.squeeze(0)

    if keep_track:
        return noisy, all_noise

    return noisy