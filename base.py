import os,yaml
import torch
#torch.set_float32_matmul_precision('medium')
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
from pathlib import Path
from typing import Optional

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist

from model import SwinTransformerFineTune
from data import list_of_bags_collate_fn


## Iska istamal hoga bhi ya nahi? Koi idea nahi hai
## dekte hain
## Added 28-Jan-2025
def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize the process group for DDP.
    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        backend (str): Backend for distributed training ('nccl' for GPU, 'gloo' for CPU).
    """
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"[RANK {rank}] Process group initialized, World Size: {world_size}")
    torch.cuda.set_device(rank)

## Added 28-Jan-2025
def cleanup_ddp():
    """Clean up the process group."""
    dist.destroy_process_group()

class SwinLightningModule(LightningModule):
    def __init__(self, model, loss_func, train_config):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.train_config = train_config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        #inputs, labels = inputs.to(self.device), labels.to(self.device) ## labels list hai. Yeh like error degi
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels)
        ## Added on 29-Jan-2025
        ## Loss per GPU
        # gathered_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_losses, loss)
        # print(f"[Rank {dist.get_rank()}] Losses across GPUs: {gathered_losses}")
        ### End of addition
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.train_config['BASE_LR']),
            weight_decay=float(self.train_config['WEIGHT_DECAY'])
        )
        total_steps = self.train_config['EPOCHS'] * (self.train_config['STEPS_PER_EPOCH'] // self.train_config.get('GRAD_ACCUM', 1))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=float(self.train_config['BASE_LR']),
        #     total_steps=total_steps,
        #     pct_start=self.train_config['WARMUP_EPOCHS'] / self.train_config['EPOCHS'],
        #     anneal_strategy='cos',
        #     final_div_factor=1e4
        # )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

def train(
    train_datasets, test_datasets,
    config_path: str,
    path: Optional[Path] = None,
    device: Optional[torch.device] = None
):
    """
    Train the SwinTransformerFineTune model using PyTorch Lightning.
    Args:
        train_datasets: PyTorch Dataset for training.
        test_datasets: PyTorch Dataset for validation.
        config_path: Path to the YAML configuration file.
        path: Path to save logs and models.
        device: Device for training (CPU/GPU).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['MODEL']
    train_config = config['TRAIN']

    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1)) ## defaulting to 0 if I want to work on single GPU

    distributed = world_size > 1 # Agar word_size >1, toh DDP enabled hai, assume karle didi
    # Return mein mujhe Boolean milage. Agar 1 se zyada hai toh True, nahi toh False

    if device is None:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    print('Distributed:', distributed)
    if distributed:
        setup_ddp(rank, world_size)

    # Initialize W&B logger
    wandb_logger = WandbLogger(project=model_config['NAME'])

    pin_memory = torch.cuda.is_available()

    train_sampler = DistributedSampler(train_datasets["train"], 
                                       num_replicas=world_size,
                                       rank=rank) if distributed else None
    
    valid_sampler = DistributedSampler(
        test_datasets["test"],
        num_replicas=world_size,
        rank=rank,
        shuffle=False) if distributed else None

    train_dl = DataLoader(
        train_datasets["train"],
        sampler = train_sampler,
        batch_size=train_config['BATCH_SIZE'], 
        collate_fn=list_of_bags_collate_fn,
        shuffle=not distributed, ## Shuffle tabi karna hai jab distributed nahi hai, 
        ## kyunki tabi pata rahega ki saara data use ho raha hai har process mein. 
        ## Also har epoch mein wohi data split use hoga
        num_workers=train_config['NUM_WORKERS'],
        pin_memory=pin_memory
    )

    valid_dl = DataLoader(
        test_datasets["test"],
        sampler=valid_sampler,
        batch_size=train_config['VALID_BATCH_SIZE'], 
        shuffle=False,
        collate_fn=list_of_bags_collate_fn,
        num_workers=train_config['NUM_WORKERS'],
        pin_memory=pin_memory,
        #drop_last=False # Iski zarurat nahi hai, kyunki hum waise bhi saare patches le rahen hai.
    )

    train_config['STEPS_PER_EPOCH'] = len(train_dl)

    # Calculate and log class weights
    counts = pd.Series(train_datasets["train"].sample_to_class).value_counts()
    weight = counts.sum() / counts
    weight /= weight.sum()
    class_weights = torch.tensor(weight.values, dtype=torch.float32, device=device)

    model = SwinTransformerFineTune(
        num_classes=train_config.get('NUM_CLASSES', 2),
        hidden_dim1=model_config['EMBED_DIM'],
        hidden_dim2=model_config.get('HIDDEN_DIM2', 512),
        dropout_prob=model_config.get('DROP_PATH_RATE', 0.3)
    ).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank
        )

    lightning_model = SwinLightningModule(
        model=model, 
        loss_func=nn.CrossEntropyLoss(weight=class_weights), 
        train_config=train_config
    )

    # Set distributed training options
    devices = torch.cuda.device_count() if distributed else 1
    strategy = 'ddp' if distributed and devices > 1 else 'auto'

    trainer = Trainer(
        logger=wandb_logger,  # Use W&B logger
        max_epochs=train_config['EPOCHS'],
        accumulate_grad_batches=train_config.get('GRAD_ACCUM', 1),
        strategy=strategy,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        precision="16-mixed",  # Enables mixed precision
        num_sanity_val_steps=0,  # Disable sanity checks
        val_check_interval=1.0,
        #limit_val_batches=1.0, # Iska kya matlab hai??
        #enable_progress_bar=int(os.getenv('RANK', 0)) == 0,
        enable_progress_bar=True,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', dirpath=path, filename='best_valid', save_top_k=1, mode='min'),
            EarlyStopping(monitor='val_loss', patience=train_config['PATIENCE'], mode='min'),
            LearningRateMonitor(logging_interval='step')
        ],
        log_every_n_steps=1,
    )

    trainer.fit(lightning_model, train_dl, valid_dl)

    if distributed:
         cleanup_ddp()

    return trainer