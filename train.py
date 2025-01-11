import pytorch_lightning as pl
import wandb
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader

from src.dataset import PatentDataset
from src.model import PatentClassifier

def main():

    # Set seed for reproducibility
    seed = 1999
    pl.seed_everything(seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load tokenizer and base_model
    model_name = "anferico/bert-for-patents"
    num_labels = 4
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    base_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Create dataset 
    train_ds = PatentDataset('/home/fantoni/patent-sentence-classification/data/train.xlsx', tokenizer=bert_tokenizer)
    eval_ds = PatentDataset('/home/fantoni/patent-sentence-classification/data/eval.xlsx', tokenizer=bert_tokenizer)
    test_ds = PatentDataset('/home/fantoni/patent-sentence-classification/data/test.xlsx', tokenizer=bert_tokenizer)
    print(f"Train set size: {len(train_ds)}")
    print(f"Validation set size: {len(eval_ds)}")
    print(f"Test set size: {len(test_ds)}")

    # Create dataloader
    batch_size = 4
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size)

    # Setup wandb logger
    wandb_logger = WandbLogger(project='patent-sentence-classification')

    # Setup call back for Trainer
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='/home/fantoni/patent-sentence-classification/models',
            filename='best-checkpoint',
            save_top_k=1, # if 1, saves only the best checkpoint based on the monitored metric.
            verbose=True,
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min', # stop training when the monitored metric stops decreasing 
            patience=3, # the training will continue for up to 3 epochs without improvement in the monitored metric before stopping.
            verbose=True
        )
    ]

    # Set up Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=accelerator,
        devices= 1 if accelerator == "gpu" else None,
        max_epochs=3,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        val_check_interval=0.1,  # if set to 1, validate only once per epoch
        #limit_train_batches=0.01,  # Use only 1% of training data
        #limit_val_batches=0.01,    # Use only 1% of validation data
        check_val_every_n_epoch=1
    )

    # Freeze BERT base model parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # Unfreeze classifier head parameters
    for param in base_model.classifier.parameters():
        param.requires_grad = True

    # Convert BERT base model to Lightning module
    model = PatentClassifier(model=base_model, tokenizer=bert_tokenizer).to(device)

    # Perform Train and Test
    trainer.fit(model, train_dl, eval_dl)
    trainer.test(model, test_dl)

    # Terminate run on wandb
    wandb.finish() 
    print('Training and testing completed.')

if __name__ == "__main__":
    main()
    