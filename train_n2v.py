import argparse
import logging as log
import careamics as cm
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from careamics import CAREamicsModuleWrapper
from careamics import CAREamist
from careamics.utils.metrics import psnr
from careamics.lightning_datamodule import TrainingDataWrapper
from careamics.lightning_prediction_loop import CAREamicsPredictionLoop

from datasets import load_split_datasets
import os

# CPU COUNT
log.basicConfig(level=log.INFO)

def main(dataset_name, batch_size=256, take_n=-1, use_n2v2=False, output_root='models/', epochs=1000, input_channel=None):
    """
    Train a Noise2Void model on the given dataset.

    Args:
    dataset_name (str): Name of the dataset to use.
    batch_size (int): Batch size for training.
    take_n (int): Number of images to take from the dataset. Default is -1, which means all images.
    use_n2v2 (bool): Use Noise2Void2 model.
    output_root (str): Output ROOT directory for the model.
    epochs (int): Number of epochs to train the model.
    input_channel (int): Input channel to use. Default is None, which means all channels.
    """

    model_name = 'n2v2'if use_n2v2 else 'n2v'

    log.info(f"Training Noise2Void model on dataset {dataset_name} with batch size {batch_size} and model {model_name}.")
    log.info(f"Model will be saved to {output_root}")

    if take_n > 0:
        log.warning(f"Taking only the first {take_n} samples from the dataset.")

    (train, val), (train_mean, train_std), (val_mean, val_std) = load_split_datasets(dataset_name, channelwise_stats=(input_channel is not None))
    
    if take_n > 0:
        train = train[:take_n]
        val = val[:take_n]
    
    if len(train.shape) == 3:
        log.warning(f"Dataset has only one channel. Using only channel 0.")
        train = train[:, np.newaxis, :, :]
        val = val[:, np.newaxis, :, :]
        input_channel = 0

    if input_channel is not None:
        log.info(f"Using only channel {input_channel} for training.")
        train = train[:, input_channel]
        val = val[:, input_channel]
    else:
        #UNWRAP CHANNELS
        log.info(f"Unwrapping channels...")
        train = train.reshape(-1, train.shape[-2], train.shape[-1])
        val = val.reshape(-1, val.shape[-2], val.shape[-1])

    log.info(f"Got train shape: {train.shape}")
    log.info(f"Got val shape: {val.shape}")
    # Create the model
    model = CAREamicsModuleWrapper(algorithm="n2v",
                                loss="n2v",
                                architecture="UNet",
                                model_parameters={"n2v2": use_n2v2},
                                optimizer_parameters={"lr": 0.5e-3},
                                lr_scheduler_parameters={"factor": 0.5, "patience": 10},
                                )


    train_data_module = TrainingDataWrapper(
                                        batch_size=batch_size,
                                        train_data=train,
                                        val_data=val,
                                        data_type="array",
                                        patch_size=(64,64),
                                        axes="SYX",
                                        use_n2v2=use_n2v2,
                                        struct_n2v_axis="none",
                                        struct_n2v_span=7,
                                        dataloader_params={"num_workers": os.cpu_count()},
                                        )
    

    output_dir = os.path.join(output_root, 
                              dataset_name, 
                              'ch_all' if input_channel is None else f'ch_{input_channel}', 
                              model_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        mode='min'
    )

    # class SavePredictions(Callback):
    #     def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    #         log.info("Saving predictions...")
    #         tiled_loop = CAREamicsPredictionLoop(trainer)
    #         trainer.predict_loop = tiled_loop
    #         preds = trainer.predict(model, datamodule=train_data_module)
    #         log.info("Predictions saved.")

    # Save some predictions when the model is saved
    # save_predictions_callback = SavePredictions()


    trainer = Trainer(max_epochs=epochs, default_root_dir=output_dir, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=train_data_module)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='AI4Life Dataset Handler')
    argparser.add_argument('--dataset_name', type=str, help='Name of the dataset to use.')
    argparser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    argparser.add_argument('--take_n', type=int, default=-1, help='Number of images to take from the dataset. Default is -1, which means all images.')
    argparser.add_argument('--use_n2v2', action='store_true', help='Use Noise2Void2 model.')
    argparser.add_argument('--output_dir', type=str, default='models/', help='Output ROOT directory for the model.') 
    argparser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train the model.')
    argparser.add_argument('--input_channel', type=int, default=None, help='Input channel to use. Default is None, which means all channels.')
    args = argparser.parse_args()

    main(args.dataset_name, args.batch_size, args.take_n, args.use_n2v2, output_root=args.output_dir, epochs=args.epochs, input_channel=args.input_channel)