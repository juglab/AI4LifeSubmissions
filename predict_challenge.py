from careamics import CAREamist
from datasets import load_split_datasets, load_datasets_yml, iter_tiff_batch
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import argparse
import glob


def predict_n2v(input_path:str, model_ckpt:str, batch_size: str):
    all_files = glob.glob(os.path.join(input_path, '*.tif*'))
    model = CAREamist(model_ckpt, work_dir=os.path.dirname(os.path.dirname(model_ckpt)))

    for tiff_in_path in all_files:
        if 'unstructured-noise' in tiff_in_path:
            tiff_out_path = tiff_in_path.replace('unstructured-noise', 'denoised')
        elif 'structured-noise' in tiff_in_path:
            tiff_out_path = tiff_in_path.replace('structured-noise', 'denoised')
        else:
            raise ValueError(f"Path {tiff_in_path} does not contain the correct patern 'image-stack-[un]structured-noise'") 

        tiff_input = tifffile.imread(tiff_in_path)
        tiff_input = tiff_input[None, ...]
        print(tiff_input.shape)
        pred_batch = model.predict(source=tiff_input, data_type='array', axes='SCYX' if tiff_input.ndim == 4 else 'SYX')
        pred_batch = pred_batch.squeeze()
        print(pred_batch.shape)
        os.makedirs(os.path.dirname(tiff_out_path), exist_ok=True)
        tifffile.imwrite(tiff_out_path, pred_batch)

def predict_challenge(input_path: str, model_ckpt:str, model_name: str, batch_size: int):
    if model_name.lower() in ['n2v', 'n2v2']:
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
    
    
    