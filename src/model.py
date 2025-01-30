import torch
import pytorch_lightning as pl
from torchmetrics import Precision, Recall, F1Score

from src.config import load_config

class PatentSentenceClassifier(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.cfg = load_config()
        self.model = model
        self.tokenizer = tokenizer

        # Initialize metrics for each stage
        metrics = {
            'precision': Precision(task='multiclass', num_classes=self.cfg.model.num_lables, average='weighted'),
            'recall': Recall(task='multiclass', num_classes=self.cfg.model.num_lables, average='weighted'),
            'f1': F1Score(task='multiclass', num_classes=self.cfg.model.num_lables, average='weighted'),
        }
        
        # Create metric instances for each stage
        self.train_metrics = torch.nn.ModuleDict({f"train_{k}": v.clone() for k, v in metrics.items()})
        self.val_metrics = torch.nn.ModuleDict({f"val_{k}": v.clone() for k, v in metrics.items()})
        self.test_metrics = torch.nn.ModuleDict({f"test_{k}": v.clone() for k, v in metrics.items()})
    
    def _compute_metrics(self, metrics, preds, labels):
        for name, metric in metrics.items():
            value = metric(preds, labels)
            self.log(name, value, on_step=False, on_epoch=True)

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log('train_loss', outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        self._compute_metrics(self.val_metrics, preds, batch['labels'])
        self.log('val_loss', outputs.loss, prog_bar=True)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        self._compute_metrics(self.test_metrics, preds, batch['labels'])
        self.log('test_loss', outputs.loss, prog_bar=True)
        return outputs.loss
    
    #def configure_optimizers(self):
    #    optimizer = getattr(torch.optim, self.cfg.optimizer.name)
    #    return optimizer(self.parameters(), **self.cfg.optimizer.args)
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.args
        )

        # Learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.cfg.scheduler.step_size, 
                gamma=self.cfg.scheduler.gamma
            ),
            'interval': 'epoch',  # Update the scheduler every epoch
            'monitor': 'val_loss'  # Monitor validation loss for scheduling
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}