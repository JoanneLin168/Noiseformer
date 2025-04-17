import os
import csv
import json
import sys
import shutil
import argparse

import torch
from torchvision import io
from tqdm import tqdm
import numpy as np

from dataset.noise import StarlightNoise

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)


def is_image(filename):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(filename.lower().endswith(ext) for ext in img_extensions)


def process_folder(folder_path, output_path, num_frames=16):

    # Only process first num_frames images
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and is_image(f)]
    image_files.sort()  # sort
    clean_frames_dir = os.path.join(output_path, "frames")
    os.makedirs(clean_frames_dir, exist_ok=True)

    # Move the first num_frames images into the frames subfolder
    for file in image_files[:num_frames]:
        src = os.path.join(folder_path, file)
        dst = os.path.join(clean_frames_dir, file)
        shutil.copy(src, dst)

    # Create subfolders for processed images ("noisy_frames") and labels ("labels")
    noisy_frames_dir = os.path.join(output_path, "noisy_frames")
    os.makedirs(noisy_frames_dir, exist_ok=True)

    # Get the noise profile for this folder
    folder_id = os.path.basename(folder_path)
    noise_params = {}
    noise_params_path = "./noise_parameters.csv"
    with open(noise_params_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['folder'] == folder_id:
                noise_params = {k: float(v) for k, v in row.items() if k != 'folder'}
                break

    # Process each image in the frames folder
    for file in os.listdir(clean_frames_dir):
        if is_image(file):

            file_path = os.path.join(clean_frames_dir, file)
            image = io.read_image(file_path) / 255.0

            # Apply brightness
            alpha = noise_params["alpha_brightness"]
            gamma = noise_params["gamma_brightness"]
            image = alpha*(np.power(image, 1/gamma)) # NOTE: no need for eps as create_random_noises.py doesn't allow gamma=0

            # Save low-light clean image
            io.write_png((255 * image).to(torch.uint8), file_path)

            noisy_image = StarlightNoise(image, noise_params, device="cpu")
            noisy_image = (255 * noisy_image).to(torch.uint8)
            
            # Save processed tensor and corresponding label
            base_filename, _ = os.path.splitext(file)
            noisy_frame_path = os.path.join(noisy_frames_dir, base_filename + "_noisy.png")

            io.write_png(noisy_image, noisy_frame_path)

        else:
            print(f"Skipping file: {file}")
        
    label_filepath = os.path.join(output_path, "labels.json")
    with open(label_filepath, "w") as f:
        json.dump(noise_params, f)


def main(args):
    input_path = args.input

    # Iterate over each folder in the provided input_path
    for item in tqdm(os.listdir(input_path)):
        sys.stdout.write(f"\rProcessing folder: {item}")
        sys.stdout.flush()
        folder_path = os.path.join(input_path, item)
        output_path = os.path.join(args.output, item)
        if os.path.isdir(folder_path):
            process_folder(folder_path, output_path, args.num_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to the validation set folder")
    parser.add_argument("-o", "--output", help="Path to store the processed images")
    parser.add_argument("-n", "--num_frames", type=int, default=16, help="Number of frames to use in each folder")
    args = parser.parse_args()

    main(args)