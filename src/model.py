import torch
import pytorch_lightning as pl
#from transformers import AdamW

from src.config import load_config

class PatentClassifier(pl.LightningModule):
    def __init__(self, model, tokenizer, learning_rate=2e-5):
        super().__init__()
        self.cfg = load_config()
        self.model = model
        self.tokenizer = tokenizer
        #self.learning_rate = learning_rate

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log('train_loss', outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        accuracy = (preds == batch['labels']).float().mean()
        self.log('val_loss', outputs.loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        accuracy = (preds == batch['labels']).float().mean()
        self.log('test_loss', outputs.loss, prog_bar=True)
        self.log('test_accuracy', accuracy, prog_bar=True)
        return outputs.loss
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer.name)
        return optimizer(self.parameters(), **self.cfg.optimizer.args)

    #def configure_optimizers(self):
    #  optimizer = AdamW(self.parameters(), lr=self.learning_rate)
    #  return optimizer