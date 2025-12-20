# TinyCodes Training Configuration
# ---------------------------------
# Train SpinNet on Python code snippets

# Output
out_dir = 'experiments/out-tinycodes'

# Data
dataset = 'tinycodes'
block_size = 256
batch_size = 32
gradient_accumulation_steps = 4

# Model (same as TinyStories for comparison)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# Training
max_iters = 5000
eval_interval = 200
eval_iters = 100
log_interval = 20

# Optimizer
learning_rate = 3e-3
min_lr = 6e-5
warmup_iters = 500
lr_decay_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# System
device = 'cuda'
compile = False  # Compile can cause issues with some ops
dtype = 'bfloat16'

# SpinNet-specific
octonion_attention = True  # Enable octonion head mixing
