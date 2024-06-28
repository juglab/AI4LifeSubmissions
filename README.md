# AI4Life N2V, N2V2 and HDN SUMBISSIONS

### Dataset info
````
jump_cell_painting train: train_shape=(413, 4, 540, 540)
jump_cell_painting val: val_shape=(104, 4, 540, 540)
w2s train: train_shape=(96, 3, 512, 512)
w2s val: val_shape=(24, 3, 512, 512)
hagen train: train_shape=(63, 1024, 1024)
hagen val: val_shape=(16, 1024, 1024)
support train: train_shape=(800, 1024, 1024)
support val: val_shape=(201, 1024, 1024)

````

## Steps to reproduce

## 1. Download and split data

Dataset urls and save paths are defined in `datasets.yml`.

``` python datasets.py --split --split_ratio=0.8 --seed=1234567890 ```



### N2V EXPERIMENT 1
 Trained N2V on every dataset, considering channels as individual images.

 #### Parmeters:
````bash
   python train_n2v.py --epochs 180 --output_dir models/run1 --dataset_name jump_cell_painting
   python train_n2v.py --epochs 180 --output_dir models/run1 --dataset_name w2s
   python train_n2v.py --epochs 180 --output_dir models/run1 --dataset_name hagen
   python train_n2v.py --epochs 180 --output_dir models/run1 --dataset_name support
````

#### Results:

| jump_cell_painting | w2s | hagen | support |
| --- | --- | --- | --- |
| <img src="models/run1/jump_cell_painting/n2v/lightning_logs/version_1/loss.png" width="400"> | <img src="models/run1/w2s/n2v/lightning_logs/version_1/loss.png" width="400"> | <img src="models/run1/hagen/n2v/lightning_logs/version_1/loss.png" width="400"> | <img src="models/run1/support/n2v/lightning_logs/version_1/loss.png" width="400"> |
| Likely underfitting | Overfitting | Almost ok (LR too high) | Overfitting | 

#### Hints
- Hagen is the only dataset that trains correctly, according to the loss...
- [ ] Try channelwise normalization before training (if using a single model for all channels)
- [ ] Check the dataset (patch size?)
- [ ] Try reducing the learning rate
- [ ] Should we save the best PSNR model (and the last)? or validation loss is enough?
- [x] Try learning channel-wise

### RUN 2

#### Parameters:
```bash
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name jump_cell_painting --input_channel 0
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name jump_cell_painting --input_channel 1
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name jump_cell_painting --input_channel 2
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name jump_cell_painting --input_channel 3
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name w2s --input_channel 0
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name w2s --input_channel 1
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name w2s --input_channel 2
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name hagen --input_channel 0
   python train_n2v.py --epochs 500 --output_dir models/run2 --dataset_name support --input_channel 0
```


- Training is now channel-wise, so we have a model for each channel for multichannel datasets (jump_cell and w2s)
- Learning rate is reduced to 0.5e-3 (halved)
- Models are running for 500 epochs
