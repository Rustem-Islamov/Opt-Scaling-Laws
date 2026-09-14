from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelHyperparameters:
    # Data
    input_bin: str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin: str = 'data/fineweb10B/fineweb_val_*.bin'

    # Optimization
    batch_size: int = 8*64
    device_batch_size: int = 64
    sequence_length: int = 1024
    num_iterations: int = 5100
    warmup_iters: int = 0
    warmdown_iters: int = 1450

    # Evaluation and logging
    val_loss_every: int = 125
    val_tokens: int = 10485760
    save_every: int = 0

    # Model architecture
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

@dataclass
class ScionHyperparameters(ModelHyperparameters):
    # Scion optimizer params
    unconstrained: bool = False
    momentum: float = 0.1
    weight_decay: float = 0

    # Learning rates
    lr_embed: float = 0.00036
    lr_matrix: float = 0.00036

    # Scales
    scale_embed: float = 50
    scale_matrix: float = 3000

@dataclass
class MuonHyperparameters(ModelHyperparameters):
    # Muon
    weight_decay : float = 0
    momentum: float = 0.05
    eps: float = 1e-8

    # Learning rates
    lr_embed: float = 0.01
    lr_matrix: float = 0.02

    # Adam hyperparameters
    beta1: float = 0.9
    beta2: float = 0.95

    # Optimistic
    a: Optional[float] = 0.01

@dataclass 
class ScionTraceHyperparameters(ScionHyperparameters):
    # Tracking params
    trace_stats_every: int = 200
    trace_step_T: int = 512
    trace_step_S: int = 8
    trace_step_max: int = 16