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

from datasets import load_split_datasets, load_datasets_yml
import argparse
import logging as log
import os
from numpy.random import shuffle

get_cached_patch_path = lambda dset_basefolder, split, patch_size: os.path.join(dset_basefolder, f"hdn_patches_{split}_{patch_size}.npy")

def cache_patches(
              dataset_name: str, 
              dataset_yml: str, 
              patch_size: int = 64,
              val_data_limit: int = 1000,
              ):
    """
        Store pre-computed augmentations patches into the same folder of the provided dataset for later use.
        Saved files are already normalized. Train data is already shuffled.
    """
    log.info(f"Pre-computing patches for {dataset_name} and patch size {patch_size}")

    (train, val), (train_mean, train_std), (val_mean, val_std) = load_split_datasets(dataset_name, dataset_yml=dataset_yml)

    log.info("Applying augmentations")
    train = utils.augment_data(train)
    img_width = train.shape[-1]
    img_height = train.shape[-2]
    num_patches = int(float(img_width*img_height)/float(patch_size**2)*1)
    train_images = utils.extract_patches(train, patch_size, num_patches)
    val_images = utils.extract_patches(val, patch_size, num_patches)
    val_images = val_images[:val_data_limit] # We limit validation patches to 1000 to speed up training but it is not necessary
    test_images = val_images[:100]
    print("Shape of training images:", train_images.shape, "Shape of validation images:", val_images.shape)
    
    data_basefolder = os.path.dirname([d["path"] for d in load_datasets_yml(dataset_yml=dataset_yml) if d['name'] == dataset_name][0])
    mean_path = get_cached_patch_path(data_basefolder, 'mean', patch_size)
    var_path = get_cached_patch_path(data_basefolder, 'var', patch_size)
    train_path = get_cached_patch_path(data_basefolder, 'train', patch_size)
    val_path = get_cached_patch_path(data_basefolder, 'val', patch_size)
    test_path = get_cached_patch_path(data_basefolder, 'test', patch_size)
    all_patches = np.concatenate((train_images, val_images, test_images), axis=0)
    patches_mean = all_patches.mean(axis=((0, 2, 3) if all_patches.ndim == 4 else None), keepdims=True)
    patches_var = all_patches.mean(axis=((0, 2, 3) if all_patches.ndim == 4 else None), keepdims=True)
    del all_patches

    log.info(f"Normalizing data...")
    train_images = (train_images-patches_mean)/patches_var
    val_images = (val_images-patches_mean)/patches_var
    test_images = (test_images-patches_mean)/patches_var
    

    log.info(f"Shuffling training data")
    shuffle(train_images)
    

    log.info(f"Train patches shape: {train_images}")
    log.info(f"Val patches shape: {val_images}")  
    log.info(f"Test patches shape: {test_images}")

    np.save(mean_path, patches_mean)
    log.info(f"Patches mean saved to {mean_path}")
    np.save(var_path, patches_var)
    log.info(f"Patches variance saved to {var_path}")
    
    np.save(train_path, train_images)
    log.info(f"Train patches saved to {train_path}")
    np.save(val_path, val_images)
    log.info(f"Validation patches saved to {val_path}")
    np.save(test_path, test_images)
    log.info(f"Testing patches saved to {test_path}")


def load_cached_patches(dataset_name: str, dataset_yml: str, patch_size: int=64):
    """
        Load pre-computed patches
    """
    data_basefolder = os.path.dirname([d["path"] for d in load_datasets_yml(dataset_yml=dataset_yml) if d['name'] == dataset_name][0])
    train_path = get_cached_patch_path(data_basefolder, 'train', patch_size)
    val_path = get_cached_patch_path(data_basefolder, 'val', patch_size)
    test_path = get_cached_patch_path(data_basefolder, 'test', patch_size)
    
    train_images = np.load(train_path)
    val_images = np.load(val_path)
    test_images = np.load(test_path)

    return train_images, val_images, test_images


def train_hdn(
              output_root:str, 
              dataset_name: str, 
              dataset_yml: str, 
              noise_model: str,
              patch_size: int = 64,
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

    try:
        train_images, val_images, test_images = load_cached_patches(dataset_name=dataset_name, dataset_yml=dataset_yml, patch_size=patch_size)
    except Exception as e:
        log.error(f"Error in loaded cached patches for dataset {dataset_name}. Did you run train_hdn.py --cache_patches before?")
        return
        
    img_shape = (train_images.shape[-2], train_images.shape[-1])
  
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

    parser = argparse.ArgumentParser(description="Train an HDN model from a noisy dataset and a given ground truth generated from another model, e.g., N2V/N2V2")
    parser.add_argument('--output_root', type=str, default='models/')
    parser.add_argument('--cache_patches', action='store_true', help="Cache the datasets patches in the dataset folder for later use. This is needed to be able to train.")
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to use as observations (the full dataset will be used).')
    parser.add_argument('--dataset_yml', type=str, help='Dataset yml descriptor', default="datasets.yml"),
    parser.add_argument('--noise_model_name', type=str, help='Name of the noise model to train HDN.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch Size for both augmentation and model training.')

    args = parser.parse_args()
    if args.cache_patches:
        cache_patches(dataset_name=args.dataset_name,
                      dataset_yml=args.dataset_yml,
                      patch_size=args.patch_size,
                      val_data_limit=1000)
    else:
        train_hdn(
                    output_root=args.output_root,
                    dataset_name=args.dataset_name,
                    dataset_yml=args.dataset_yml,
                    noise_model=args.noise_model_name,
                    batch_size = args.batch_size,
                    patch_size=args.patch_size
                )
