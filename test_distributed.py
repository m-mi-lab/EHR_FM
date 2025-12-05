"""
Quick test script to verify distributed MoE implementation.
Run with: python test_distributed.py
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import GPT2Config
from omegaconf import OmegaConf

# Assuming src is in Python path
from src.model import GPT2LMNoBiasModel


def test_single_gpu():
    """Test MoE model on single GPU."""
    print("\n=== Testing Single GPU ===")
    
    # Create a small MoE config
    moe_config = OmegaConf.create({
        'n_embd': 128,
        'n_layer': 2,
        'n_head': 4,
        'n_positions': 256,
        'n_experts': 4,
        'top_k': 2,
        'bias': False,
        'resid_pdrop': 0.1,
        'embd_pdrop': 0.1,
        'attn_pdrop': 0.1,
        'activation_function': 'gelu',
        'use_aux_loss': True,
        'aux_loss_weight': 0.01,
        'use_router_z_loss': False,
        'router_z_loss_weight': 0.001,
        'train_capacity': 1.25,
        'eval_capacity': 2.0,
        'min_capacity': 4,
        'expert_distributed': False,  # Explicitly disable for single GPU test
        'allow_var_seq_len': False,
    })
    
    base_config = GPT2Config(
        vocab_size=512,
        n_positions=256,
        n_embd=128,
        n_layer=2,
        n_head=4,
        activation_function='gelu',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bias=False,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GPT2LMNoBiasModel(base_config, moe_config).to(device)
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 512, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, 512, (batch_size, seq_len)).to(device)
    
    # Training mode
    model.train()
    output = model(input_ids, labels)
    
    print(f"Loss: {output.loss.item():.4f}")
    print(f"Logits shape: {output.logits.shape}")
    if output.aux_loss is not None:
        print(f"Aux loss: {output.aux_loss.item():.4f}")
    
    # Test backward
    output.loss.backward()
    print("✓ Backward pass successful")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        output = model(input_ids[:1])
        print(f"Inference output shape: {output.logits.shape}")
    
    print("✓ Single GPU test passed!\n")
    return True


def test_distributed_worker(rank, world_size):
    """Test MoE model with distributed training."""
    # Initialize process group
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    print(f"[Rank {rank}/{world_size}] Initialized")
    
    # Create MoE config with distributed enabled
    moe_config = OmegaConf.create({
        'n_embd': 128,
        'n_layer': 2,
        'n_head': 4,
        'n_positions': 256,
        'n_experts': 4,
        'top_k': 2,
        'bias': False,
        'resid_pdrop': 0.1,
        'embd_pdrop': 0.1,
        'attn_pdrop': 0.1,
        'activation_function': 'gelu',
        'use_aux_loss': True,
        'aux_loss_weight': 0.01,
        'use_router_z_loss': False,
        'router_z_loss_weight': 0.001,
        'train_capacity': 1.25,
        'eval_capacity': 2.0,
        'min_capacity': 4,
        'expert_distributed': True,  # Enable distributed
        'allow_var_seq_len': False,
    })
    
    base_config = GPT2Config(
        vocab_size=512,
        n_positions=256,
        n_embd=128,
        n_layer=2,
        n_head=4,
        activation_function='gelu',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bias=False,
    )
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None
    
    model = GPT2LMNoBiasModel(base_config, moe_config).to(device)
    
    if rank == 0:
        print(f"[Rank {rank}] Model parameters: {model.num_parameters():,}")
        print(f"[Rank {rank}] Experts per layer: {moe_config.n_experts}")
        print(f"[Rank {rank}] Experts per GPU: ~{moe_config.n_experts // world_size}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 512, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, 512, (batch_size, seq_len)).to(device)
    
    model.train()
    output = model(input_ids, labels)
    
    if rank == 0:
        print(f"[Rank {rank}] Loss: {output.loss.item():.4f}")
        print(f"[Rank {rank}] Logits shape: {output.logits.shape}")
    
    # Test backward
    output.loss.backward()
    
    if rank == 0:
        print(f"[Rank {rank}] ✓ Backward pass successful")
    
    dist.barrier()
    
    if rank == 0:
        print(f"[Rank {rank}] ✓ Distributed test passed!")
    
    dist.destroy_process_group()


def test_distributed():
    """Run distributed test with multiple processes."""
    print("\n=== Testing Distributed Training ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping distributed test")
        return False
    
    world_size = min(torch.cuda.device_count(), 2)  # Test with 2 GPUs
    
    if world_size < 2:
        print(f"Only {world_size} GPU(s) available, skipping distributed test")
        return False
    
    print(f"Testing with {world_size} GPUs")
    
    try:
        mp.spawn(
            test_distributed_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        return True
    except Exception as e:
        print(f"❌ Distributed test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("Testing Distributed MoE Implementation")
    print("="*60)
    
    # Test single GPU first
    try:
        success_single = test_single_gpu()
    except Exception as e:
        print(f"❌ Single GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        success_single = False
    
    # Test distributed if available
    try:
        success_distributed = test_distributed()
    except Exception as e:
        print(f"❌ Distributed test setup failed: {e}")
        import traceback
        traceback.print_exc()
        success_distributed = False
    
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"  Single GPU:    {'✓ PASSED' if success_single else '❌ FAILED'}")
    print(f"  Distributed:   {'✓ PASSED' if success_distributed else '⊘ SKIPPED' if not torch.cuda.is_available() or torch.cuda.device_count() < 2 else '❌ FAILED'}")
    print("="*60)

