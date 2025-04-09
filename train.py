import pytorch_lightning as pl
import wandb
import torch
import time
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.config import load_config
from src.dataset import PatentDataset
from src.model import PatentSentenceClassifier


def main():

    # Load config file
    cfg = load_config('config.yaml')

    # Set the seed for reproducibility
    pl.seed_everything(cfg.train.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load tokenizer and base_model
    print(f"Using model: '{cfg.model.name}'")
    bert_tokenizer = BertTokenizer.from_pretrained(cfg.model.name)
    base_model = BertForSequenceClassification.from_pretrained(cfg.model.name, num_labels=cfg.model.num_lables)

    # Create dataset 
    train_ds = PatentDataset(file_path=cfg.data.train_path, tokenizer=bert_tokenizer, max_length=cfg.model.max_length)
    eval_ds = PatentDataset(file_path=cfg.data.validation_path, tokenizer=bert_tokenizer, max_length=cfg.model.max_length)
    test_ds = PatentDataset(file_path =cfg.data.test_path, tokenizer=bert_tokenizer, max_length=cfg.model.max_length)
    print(f"Train set size: {len(train_ds)}. Using: '{cfg.data.train_path}'")
    print(f"Validation set size: {len(eval_ds)}. Using: '{cfg.data.validation_path}'")
    print(f"Test set size: {len(test_ds)}. Using: '{cfg.data.test_path}'")

    # Create dataloader
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    eval_dl = DataLoader(eval_ds, batch_size=cfg.validation.batch_size, num_workers=cfg.validation.num_workers)
    test_dl = DataLoader(test_ds, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers)
    
    # Initialize wandb logger 
    wandb_logger = WandbLogger(
        project=cfg.wandb.project, # set project name
        name=cfg.train.save_name,  # Set a custom run name
        config=OmegaConf.to_container(cfg, resolve=True)  # convert config file to dict before logging to wandb
    )

    # Setup call back for Trainer
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=cfg.train.save_dir,
            filename=cfg.train.save_name,
            save_top_k=1,  # if 1, saves only the best checkpoint based on the monitored metric.
            verbose=True,
        )
    ]

    # Conditionally add EarlyStopping callback if enabled
    if cfg.validation.early_stopping:
        print(f"Early stopping is enabled. Using patience: {cfg.validation.patience}")
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                mode='min',  # stop training when the monitored metric stops decreasing 
                patience=cfg.validation.patience,  # the training will continue for up to N steps without improvement in the monitored metric before stopping.
                verbose=True
            )
        )
    else:
        print("Early stopping is disabled. Training will continue without early termination.")



    # Set up Trainer
    accelerator = "gpu" if (torch.cuda.is_available() and cfg.train.use_gpu) else "cpu"
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=accelerator,
        devices= cfg.train.gpus if accelerator == "gpu" else None,
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        val_check_interval=cfg.train.validate_every,  # if set to 1, validate only once per epoch
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    '''Uncomment the following lines to finetune only the classification head of the base model.'''
    # Freeze BERT base model parameters
    #for param in base_model.parameters():
    #    param.requires_grad = False

    # Unfreeze classifier head parameters
    #for param in base_model.classifier.parameters():
    #    param.requires_grad = True

    # Convert BERT base model to Lightning module
    model = PatentSentenceClassifier(model=base_model, tokenizer=bert_tokenizer).to(device)

    # Perform Train 
    start_time = time.time()
    trainer.fit(model, train_dl, eval_dl)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"training time: {training_time:.2f} sec")
    print(f"training time: {(training_time)/60:.2f} min")
    wandb.log({"training time (sec)": training_time})

    print(f"Model saved to: {cfg.train.save_dir}/{cfg.train.save_name}.ckpt")
    
    # Perfrom Test
    trainer.test(model, test_dl)
    
    # Terminate run on wandb
    wandb.finish() 

if __name__ == "__main__":
    main()
    