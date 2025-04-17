import sys, os
import numpy as np
import torch
from PIL import Image
from src import utils
import tqdm

import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


class Trainer():
    def __init__(self, args, model, train_set, val_set, writer):
        self.args = args
        self.folder_name = args.output_folder
        self.writer = writer
        self.display_freq = args.display_freq
        self.log_freq = args.log_freq
        self.save_freq = args.save_freq
        
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.eval_patch_size = args.eval_patch_size
        self.num_epochs = args.epochs
        self.curr_epoch = 0
        self.total_iter = 0

        self.model = model
        self.device = args.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.w1 = args.w1
        self.w2 = args.w2
        
        # Dataloaders
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers)
        
        self.val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers)
        
    def save_image(self, img, name, folder):
        os.makedirs(folder, exist_ok=True)
        file_path = folder+name
        Image.fromarray((np.clip(img,0,1) * 255).astype(np.uint8)).save(file_path)

    def train(self):
        total_loss = 0
        self.model.train()
        for i, sample in enumerate(tqdm.tqdm(self.train_loader, desc=f'Training')):
            noisy_frames = sample['noisy'].to(self.device)
            clean_frames = sample['clean'].to(self.device)
            gt_labels = sample['gt_labels'].to(self.device)

            self.optimizer.zero_grad()

            # Reshape input from [B, C, N, H, W] to [B*N, C, H, W]
            B, C, N, H, W = noisy_frames.shape
            noisy_frames = noisy_frames.view(B * N, C, H, W)
            clean_frames = clean_frames.view(B * N, C, H, W)

            # Generate a batch of images
            synth_noisy, recon_imgs, pred_labels = self.model(clean_frames, noisy_frames)

            # Patchify the images if needed
            if self.patch_size > self.eval_patch_size:
                synth_noisy = utils.split_into_patches2d(synth_noisy, self.eval_patch_size).to(self.device)
                recon_imgs = utils.split_into_patches2d(recon_imgs, self.eval_patch_size).to(self.device)
                real_noisy = utils.split_into_patches2d(noisy_frames, self.eval_patch_size).to(self.device)
                clean = utils.split_into_patches2d(clean_frames, self.eval_patch_size).to(self.device)
            else:
                real_noisy = noisy_frames
                clean = clean_frames

            # Display images during training (before FFT for visualization)
            if self.total_iter % self.display_freq == 0:
                gt_plt = real_noisy.cpu().detach()[0]
                out_plt = synth_noisy.cpu().detach()[0]
                recon_plt = recon_imgs.cpu().detach()[0]
                concatenated_images = torch.cat((gt_plt, out_plt, recon_plt), dim=2)
                self.writer.add_image(f'Train/Images_(gt - out - recon)', concatenated_images, self.total_iter)

            # Reshape output back to [B, C, N, H, W]
            synth_noisy = synth_noisy.view(B, -1, N, H, W)
            recon_imgs = recon_imgs.view(B, -1, N, H, W)
            clean = clean.view(B, -1, N, H, W)
            pred_labels = pred_labels.view(B, N, -1)

            # Apply fourier transform for loss calculation
            real_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(real_noisy, norm="ortho")))
            synth_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(synth_noisy, norm="ortho")))
  
            mlp_loss = F.mse_loss(pred_labels, gt_labels)
            recon_loss = F.l1_loss(recon_imgs, clean)
            loss = self.w1 * mlp_loss + self.w2 * recon_loss

            if self.total_iter % self.log_freq == 0:
                self.writer.add_scalar('Train/Total_Loss', np.round(loss.item(), 5), self.total_iter)
                self.writer.add_scalar('Train/MLP_Loss', np.round(mlp_loss.item(), 5), self.total_iter)
                self.writer.add_scalar('Train/Recon_Loss', np.round(recon_loss.item(), 5), self.total_iter)

            loss.backward()
            self.optimizer.step()

            self.total_iter += 1
            total_loss = loss.item()

        print('Epoch:', self.curr_epoch, 'Total Loss:', total_loss)


    def validate(self):
        kid_metric = KernelInceptionDistance().to(self.device)
        fid_metric = FrechetInceptionDistance().to(self.device)
        tot_kld = 0
        best_kld = 1e6
        self.model.eval()

        for i, sample in enumerate(tqdm.tqdm(self.val_loader, desc='Validating')):
            with torch.no_grad():

                noisy_frames = sample['noisy'].to(self.device)
                clean_frames = sample['clean'].to(self.device)

                # Reshape input from [B, C, N, H, W] to [B*N, C, H, W]
                B, C, N, H, W = noisy_frames.shape
                noisy_frames = noisy_frames.view(B * N, C, H, W)
                clean_frames = clean_frames.view(B * N, C, H, W)
                
                synth_noisy_full, recon_imgs, pred_labels = self.model(clean_frames, noisy_frames)

                # Patchify the images if needed
                if self.patch_size > self.eval_patch_size:
                    synth_noisy = utils.split_into_patches2d(synth_noisy_full, self.eval_patch_size).to(self.device)
                    real_noisy = utils.split_into_patches2d(noisy_frames, self.eval_patch_size).to(self.device)
                    clean = utils.split_into_patches2d(clean_frames, self.eval_patch_size).to(self.device)
                else:
                    synth_noisy = synth_noisy_full
                    real_noisy = noisy_frames
                    clean = clean_frames

                synth_noisemap = synth_noisy-clean
                real_noisemap = real_noisy-clean

                synth_np = (synth_noisemap.view(B, N, C, H, W)).detach().cpu().numpy()
                real_np = (real_noisemap.view(B, N, C, H, W)).detach().cpu().numpy()
                kld_val = utils.cal_kld(synth_np, real_np)

                # Calculate validation metrics
                tot_kld += kld_val
                real_int = (real_noisemap.detach() * 255).to(torch.uint8)
                synth_int = (synth_noisemap.detach() * 255).to(torch.uint8)
                fid_metric.update(real_int, real=True)
                fid_metric.update(synth_int, real=False)
                kid_metric.update(real_int, real=True)
                kid_metric.update(synth_int, real=False)

            # Reshape output back to [B, C, N, H, W]
            noisy_frames = noisy_frames.view(B, -1, N, H, W)
            clean_frames = clean_frames.view(B, -1, N, H, W)
            synth_noisy_full = synth_noisy_full.view(B, -1, N, H, W)
            recon_imgs = recon_imgs.view(B, -1, N, H, W)
            pred_labels = pred_labels.view(B, N, -1)

            # Save full images during validation (only from the first batch)
            for n in range(N):
                gt_plt = noisy_frames.cpu().detach().numpy()[0][:, n].transpose(1,2,0)[...,0:3]
                clean_plt = clean_frames.cpu().detach().numpy()[0][:, n].transpose(1,2,0)[...,0:3]
                out_plt = synth_noisy_full.cpu().detach().numpy()[0][:, n].transpose(1,2,0)[...,0:3]
                recon_plt = recon_imgs.cpu().detach().numpy()[0][:, n].transpose(1,2,0)[...,0:3]

                output_folder = self.folder_name + f'val_images/epoch{self.curr_epoch}/video_{i}/'
                img_name = f'frame_{n}.jpg'

                self.save_image(gt_plt, img_name, output_folder+'gt/')
                self.save_image(clean_plt, img_name, output_folder+'clean/')
                self.save_image(out_plt, img_name, output_folder+'pred/')
                self.save_image(recon_plt, img_name, output_folder+'recon/')


        avg_kld = tot_kld/len(self.val_loader)
        fid_score = fid_metric.compute()
        kid_score = kid_metric.compute()[0]
        self.writer.add_scalar('Validation/Average_KLD', np.round(avg_kld, 5), self.curr_epoch)
        self.writer.add_scalar('Validation/FID', np.round(fid_score.item(), 5), self.curr_epoch)
        self.writer.add_scalar('Validation/KID', np.round(kid_score.item(), 5), self.curr_epoch)
        print('Epoch:', self.curr_epoch, 'Average KLD:', avg_kld, 'FID:', fid_score.item(), 'KID:', kid_score.item())

        # Save checkpoint
        checkpoint = {
            'epoch': self.curr_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'avg_kld': avg_kld,
            'fid_score': fid_score.item(),
            'kid_score': kid_score.item()
        }

        # Save the checkpoint
        if self.total_iter % self.save_freq == 0:
            checkpoint_name = self.folder_name + f'checkpoints/checkpoint_{self.curr_epoch}.pt'
            print('Saving Checkpoint to ', checkpoint_name)
            torch.save(checkpoint, checkpoint_name)

        # Save the best checkpoint based on KLD score
        if self.curr_epoch == 0 or avg_kld < best_kld:
            best_kld = avg_kld
            best_checkpoint_name = self.folder_name + 'checkpoints/best.pt'
            print('Saving best checkpoint')
            torch.save(checkpoint, best_checkpoint_name)

        self.curr_epoch += 1
