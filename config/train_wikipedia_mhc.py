# Wikipedia training with mHC + channel specialization
# Tests multi-stream residuals and semantic channel differentiation

out_dir = 'experiments/out-wikipedia-mhc'
eval_interval = 250
log_interval = 10
eval_iters = 100
always_save_checkpoint = True
init_from = 'scratch'

# Data
dataset = 'wikipedia'
gradient_accumulation_steps = 8
batch_size = 16
block_size = 512

# Model
n_layer = 12
n_head = 32
n_embd = 256 
dropout = 0.0
bias = False
head_mixing = True
algebra = "hadamard"  # 32D Hadamard channels
hash_embeddings = False

# mHC (Manifold Hyper-Connections) - multi-stream residuals
# Paper: arxiv.org/abs/2512.24880
use_mhc = True   # Enable doubly-stochastic residual streams
n_streams = 4    # Number of parallel streams

# Channel specialization loss - encourages semantic differentiation
# Options: 'specialization', 'contextual', 'bottleneck', or None
channel_loss = 'specialization'
channel_loss_weight = 0.01

# AdamW
learning_rate = 3e-3
max_iters = 10000  # Quick run
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Schedule
decay_lr = True
warmup_iters = 200 
lr_decay_iters = 10000
min_lr = 3e-5

# Two-stage schedule (BitNet b1.58)
two_stage_schedule = True
cooldown_start = 0.5
cooldown_lr = 1e-4
stage1_wd_peak = 0.1
stage2_wd = 0.0

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False
