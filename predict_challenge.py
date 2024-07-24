from careamics import CAREamist
from datasets import load_split_datasets, load_datasets_yml, iter_tiff_batch
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import argparse
import glob
from pathlib import Path


def save_result_image_tiff(image_array: np.ndarray, result_path: Path):
    # Taken from Challenge inference.py
    print(f"Writing an image to: {result_path}")
    with tifffile.TiffWriter(result_path) as out:
        out.write(
            image_array,
            resolutionunit=2
        )

def predict_n2v(input_path:str, model_ckpt:str, batch_size: str):
    all_files = glob.glob(os.path.join(input_path, '*.tif*'))
    print(f"Found {len(all_files)} files to predict.")
    model = CAREamist(model_ckpt, work_dir=os.path.dirname(os.path.dirname(model_ckpt)))

    for tiff_in_path in all_files:
        if 'unstructured-noise' in tiff_in_path:
            tiff_out_path = tiff_in_path.replace('unstructured-noise', 'denoised')
        elif 'structured-noise' in tiff_in_path:
            tiff_out_path = tiff_in_path.replace('structured-noise', 'denoised')
        else:
            raise ValueError(f"Path {tiff_in_path} does not contain the correct patern 'image-stack-[un]structured-noise'") 
        tiff_out_path = tiff_out_path.replace("input/", "output/")
        print(f"Predicting file {tiff_in_path}...")
        tiff_input = tifffile.imread(tiff_in_path)
        print(f"File Shape: {tiff_input.shape}")
        pred_batch = list()
        print(f"Predicting one frame at a time to avoid OOMs...")
        for img in tiff_input:
            if img.ndim == 2 or img.ndim == 3:
                img = img[None, ...]
            print(f"Img Shape: {img.shape}")
            pred_batch += model.predict(source=img, data_type='array', axes='SCYX' if img.ndim == 4 else 'SYX')
        pred_batch = np.concatenate(pred_batch, axis=0)
        print(f"Final Prediction Shape: {pred_batch.shape}")
        os.makedirs(os.path.dirname(tiff_out_path), exist_ok=True)
        save_result_image_tiff(pred_batch, Path(tiff_out_path))
        print(f"Prediction written to {tiff_out_path}")
    print(f"Done")

def predict_challenge(input_path: str, model_ckpt:str, model_name: str, batch_size: int):
    if model_name.lower() in ['n2v', 'n2v2']:
        print(f"Predicting using:")
        print(f"Input: {input_path}")
        print(f"Checkpoint: {model_ckpt}")
        print(f"Model: {model_name}")
        print(f"Batch Size: {batch_size}")
        predict_n2v(input_path=input_path,
                    model_ckpt=model_ckpt,
                    batch_size=batch_size)
    else:
        raise NotImplementedError(f"Model name {model_name} not recognized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict for AI4Life Denoising Challenge")
    parser.add_argument('--input_path', type=str, help="Input folder containing .tiff images")
    parser.add_argument('--model_ckpt', type=str, help="Checkpoint of the model to use")
    parser.add_argument('--model_name', type=str, help="Name of the model to use, it is used to choose implementation. Can be either 'n2v', 'n2v2', 'hdn'")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size to use")
    
    args = parser.parse_args()

    predict_challenge(input_path=args.input_path,
                      model_ckpt=args.model_ckpt,
                      model_name=args.model_name,
                      batch_size=args.batch_size)
    
    
    