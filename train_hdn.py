import warnings
warnings.filterwarnings('ignore')
# We import all our dependencies.
import numpy as np
import torch
import sys
sys.path.append('hdn')
from hdn.models.lvae import LadderVAE
from hdn.lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
import hdn.boilerplate.boilerplate as boilerplate
import hdn.utils as utils
from hdn import training
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm

from datasets import load_split_datasets
import argparse
import logging as log
import os


def train_hdn(
              output_root:str, 
              dataset_name: str, 
              dataset_yml: str, 
              noise_model: str,
              patch_size: int = 64,
              val_data_limit: int = 1000,
              batch_size: int = 64,
              ):
    
    """
        Trains an HDN model on the provided dataset and using the given noise model.

        Args:
            - output_root:
            - dataset_name:
            - dataset_yml:
            - noise_model: 
            - patch_size: int (Default: 64)
                We extract overlapping patches of size ```patch_size x patch_size``` from training and validation images.
                Usually 64x64 patches work well for most microscopy datasets
            - val_data_limit: int (Default: 1000)
                Limit validation data samples to speedup training.
            - batch_size: int

    """
    
    log.info(f"Setting up training of HDN for {dataset_name} using the {noise_model} noise model.")
    
    model_name = f"hdn_{noise_model}_noisemodel"
    directory_path = os.path.join(output_root, dataset_name, 'hdn')
    log.info(f"Model name: {noise_model} \n Output path: {directory_path}")

    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    (train, val), (train_mean, train_std), (val_mean, val_std) = load_split_datasets(dataset_name, dataset_yml=dataset_yml)
    log.info("Applying augmentations")

    train = utils.augment_data(train)
     
    img_width = train.shape[2]
    img_height = train.shape[1]
    num_patches = int(float(img_width*img_height)/float(patch_size**2)*1)
    train_images = utils.extract_patches(train, patch_size, num_patches)
    val_images = utils.extract_patches(val, patch_size, num_patches)
    val_images = val_images[:val_data_limit] # We limit validation patches to 1000 to speed up training but it is not necessary
    test_images = val_images[:100]
    img_shape = (train_images.shape[1], train_images.shape[2])
    print("Shape of training images:", train_images.shape, "Shape of validation images:", val_images.shape)

    # Data-specific
    gaussian_noise_std = None
    noise_model_path = f"noise_models/{dataset_name}/{noise_model}/GMM.npz"
    log.info(f"Loading noise model from {noise_model_path}")
    noise_model_params= np.load(noise_model_path)
    noiseModel = GaussianMixtureNoiseModel(params = noise_model_params, device = device)

    # Training-specific
    batch_size=64
    virtual_batch = 8
    lr=3e-4
    max_epochs = 500
    steps_per_epoch = 400
    test_batch_size=100

    # Model-specific
    num_latents = 6
    z_dims = [32]*int(num_latents)
    blocks_per_layer = 5
    batchnorm = True
    free_bits = 1.0
    use_uncond_mode_at=[0,1]

    train_loader, val_loader, test_loader, data_mean, data_std = boilerplate._make_datamanager(train_images,val_images,
                                                                                           test_images,batch_size,
                                                                                           test_batch_size)

    model = LadderVAE(z_dims=z_dims,
                    blocks_per_layer=blocks_per_layer,
                    data_mean=data_mean,
                    data_std=data_std,
                    noiseModel=noiseModel,
                    device=device,
                    batchnorm=batchnorm,
                    free_bits=free_bits,
                    img_shape=img_shape,
                    use_uncond_mode_at=use_uncond_mode_at).cuda()

    model.train() # Model set in training mode

    training.train_network(model=model,
                        lr=lr,
                        max_epochs=max_epochs,
                        steps_per_epoch=steps_per_epoch,
                            directory_path= directory_path,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            virtual_batch=virtual_batch,
                            gaussian_noise_std=gaussian_noise_std,
                            model_name=model_name, 
                            val_loss_patience=30)



if __name__ == "__main__":
    log.basicConfig()
    log.getLogger().setLevel(log.INFO)

    parser = argparse.ArgumentParser(description="Generate a Noise Model from a noisy dataset and a given ground truth generated from another model, e.g., N2V/N2V2")
    parser.add_argument('--output_root', type=str, default='models/')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to use as observations (the full dataset will be used).')
    parser.add_argument('--dataset_yml', type=str, help='Dataset yml descriptor', default="datasets.yml"),
    parser.add_argument('--noise_model_name', type=str, help='Name of the noise model to train HDN.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')

    args = parser.parse_args()

    train_hdn(
                output_root=args.output_root,
                dataset_name=args.dataset_name,
                dataset_yml=args.dataset_yml,
                noise_model=args.noise_model_name,
                batch_size = args.batch_size
            )
