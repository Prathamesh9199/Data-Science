from dataclasses import dataclass

@dataclass
class SLMConfig:
    # == Model ==
    vocab_size:         int = 32_000
    d_model:            int = 768                                           # Size of vector representing each token
    n_layers:           int = 24                                            # Number of transformer blocks
    n_heads:            int = 12                                            # Number of query heads
    n_kv_heads:         int = 4                                             # Number of key, value heads (Group Query Attention)
    ffn_dim:            int = 2048                                          # Size of hidden layer in FeedForward network
    max_seq_len:        int = 4096                                          # Maximum context window
    rope_theta:         float = 500_000.0                                   # Base frequency for rotary embeddings
    norm_eps:           float = 1e-5

    # == Dataset ==
    test_mode:          bool = True
    target_tokens:      int = 1_953_000_000                                 # 20M tokens
    validation_tokens:  int = 9_760_000                                     # 5M held-out for validation
    hf_token:           str = '<HF_TOKEN>'
    # dataset:            str = 'HuggingFaceTB/dclm-edu'
    dataset:            str = 'roneneldan/TinyStories'
    tokenizer:          str = 'mistralai/Mistral-7B-v0.1'
    train_data:         str = 'train.bin'
    validation_data:    str = 'val.bin'
    disk_batch_size:    int = 100_000_000                                   # 100M tokens per batch
    batched_dir:        str = 'batches'

    # == Training hyperparameters ==
    micro_batch:        int = 1                                             # Sequences per GPU step
    grad_accum:         int = 96                                            # Steps before optimizer update | effective batch = 1 × 96 × 4096 = 786,432 tokens/step
    lr_muon:            float = 0.02                                        # Muon peak LR (matrix weights)
    lr_adam:            float = 3e-4                                        # AdamW peak LR (embeddings, norms, biases)
    lr_min_ratio:       float = 0.1                                         # Cosine decays to 10% of peak
    warmup_steps:       int = 100                                           # Linear warmup
    weight_decay:       float = 0.1
    grad_clip:          float = 1.0
    save_every:         int = 100                                           # Save a checkpoint every N global steps
    keep_ckpts:         int = 5                                             # Rolling window — delete oldest beyond this
    log_every:          int = 10                                            # Print + write CSV every N global steps
    ckpt_dir:           str = 'checkpoints'
    log_file:           str = 'training_log.csv'

    # Thermal management
    gpu_temp_threshold: int = 85                                            # pause if GPU hits this (°C)
    cooling_break_s:    int = 600                                           # 10 min cool-down
    break_every_steps:  int = 2000                                          # mandatory break regardless of temp