import torch
from hdn.lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from hdn.lib import histNoiseModel
from hdn.lib.utils import plotProbabilityDistribution
from datasets import load_datasets_yml
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging as log

log.basicConfig()
log.getLogger().setLevel(log.INFO)



def generate_noise_model(
                            output_root:str, 
                            gt_path:str, 
                            gt_name:str, 
                            dataset_name: str, 
                            dataset_yml: str,
                            n_gaussian: int = 3,
                            n_coeff: int = 2,
                            bins: int = 250,
                            signal_bin_index: int = 100,
                            ):

    device = torch.device("cuda:0")

    dset = [d for d in load_datasets_yml(dataset_yml=dataset_yml) if d['name'] == dataset_name][0]

    out_root = os.path.join(output_root, dataset_name, gt_name)
    gt_path = gt_path or f'predictions/{dataset_name}/{gt_name}.tiff'
    
    log.info(f"Loading signals...")
    observations = tifffile.imread(dset['path'])
    datamin, datamax = observations.min(), observations.max()
    signal = tifffile.imread(gt_path).squeeze()
    log.info(f"Done.")

    try:
        log.info(f"Generating histogram...")
        histogram = histNoiseModel.createHistogram(bins=bins, minVal=datamin, maxVal=datamax, observation=observations, signal=signal)
        os.makedirs(out_root, exist_ok=True)
        np.save(os.path.join(out_root, 'histogram.npy'), histogram)
        histogramFD = histogram[0]

        # Let's look at the histogram-based noise model.
        plt.xlabel('Observation Bin')
        plt.ylabel('Signal Bin')
        plt.imshow(histogramFD**0.25, cmap='gray')
        plt.show()
        min_signal=np.percentile(signal, 0.5)
        max_signal=np.percentile(signal, 99.5)
        log.info(f"Minimum Signal Intensity is {min_signal}")
        log.info(f"Maximum Signal Intensity is {max_signal}")

        log.info(f"Training GMM for {dataset_name} with {gt_path} GT:")
        gaussianMixtureNoiseModel = GaussianMixtureNoiseModel(min_signal = min_signal, max_signal = max_signal, path=out_root+'/', weight = None, n_gaussian = n_gaussian, n_coeff = n_coeff, device = device, min_sigma = 50)
        gaussianMixtureNoiseModel.train(signal, observations, batchSize = 250000, n_epochs = 2000, learning_rate = 0.1, name = 'GMM', lowerClip = 0.5, upperClip = 99.5)
        log.info(f"Model saved Successfully")
        plotProbabilityDistribution(signalBinIndex=signal_bin_index, histogram=histogramFD, gaussianMixtureNoiseModel=gaussianMixtureNoiseModel, min_signal=datamin, max_signal=datamax, n_bin= bins, device=device)
        plt.savefig(os.path.join(out_root, 'GMM.png'))
        log.info("Figure saved to " + os.path.join(out_root, 'GMM.png'))

    except Exception as e:
        log.error(f"ERROR PROCESSING DATASET {dataset_name} and model {gt_name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Noise Model from a noisy dataset and a given ground truth generated from another model, e.g., N2V/N2V2")
    parser.add_argument('--output_root', type=str, default="noise_models/", help='Output root folder to store noise models. Each Noise model will be saved in <output_root>/<dataset_name>/<gt_name>/[GMM.npz|histogram.npy]')
    parser.add_argument('--gt_name', type=str, help='Name of the GT used to learn the model. If bootstrapping, this could be the name of the model used to generate the GTs. This is used to build the save path for noise models.')
    parser.add_argument('--gt_path', type=str, help='Path of the TIFF ground truth (or upstream model predictions, if bootstrapping) to use for generating the noise model. If None, the default is predictions/<dataset_name>/<gt_name>.tiff')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to use as observations (the full dataset will be used).')
    parser.add_argument('--dataset_yml', type=str, help='Dataset yml descriptor', default="datasets.yml")
    parser.add_argument('--n_gaussian', type=int, default=3, help='Number of gaussians to use for Gaussian Mixture Model')
    parser.add_argument('--n_coeff', type=int, default=2, help='No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components.')
    parser.add_argument('--bins', type=int, default=250, help='Number of bins to use for geneating the histogram.')
    parser.add_argument('--signal_bin_index', type=int, default=100, help='Bin index of the signal that is shown in the saved PNG image')

    args = parser.parse_args()
    generate_noise_model(output_root=args.output_root,
                         gt_path=args.gt_path,
                         gt_name=args.gt_name,
                         dataset_name=args.dataset_name,
                         dataset_yml=args.dataset_yml
                        )