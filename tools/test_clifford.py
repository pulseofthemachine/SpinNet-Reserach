#!/usr/bin/env python
"""
Test suite for Clifford/Hadamard 32D algebra and FHT kernel.

Run with: python tools/test_clifford.py
"""

import torch
import math
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fht_correctness():
    """Verify FHT butterfly matches matrix FHT reference."""
    from src.model.fht_cuda import fht_butterfly, fht_reference
    
    print("Testing FHT correctness...")
    x = torch.randn(4, 16, 32)
    
    y_butterfly = fht_butterfly(x)
    y_reference = fht_reference(x)
    
    diff = (y_butterfly - y_reference).abs().max().item()
    assert diff < 1e-5, f"FHT mismatch: {diff:.2e}"
    print(f"  ✓ Max diff: {diff:.2e}")


def test_hadamard_orthonormality():
    """Verify Hadamard matrix is orthonormal."""
    from src.model.fht_cuda import hadamard_matrix
    
    print("Testing Hadamard orthonormality...")
    H = hadamard_matrix(32, normalize=True)
    
    identity = H @ H.T
    expected = torch.eye(32)
    
    diff = (identity - expected).abs().max().item()
    assert diff < 1e-5, f"H @ H.T not identity: {diff:.2e}"
    print(f"  ✓ ||H @ H.T - I||_max: {diff:.2e}")


def test_fht_self_inverse():
    """Verify FHT is self-inverse: FHT(FHT(x)) = x."""
    from src.model.fht_cuda import fht_butterfly
    
    print("Testing FHT self-inverse property...")
    x = torch.randn(2, 8, 32)
    
    # FHT is H/sqrt(n) @ x, so FHT(FHT(x)) = H/sqrt(n) @ H/sqrt(n) @ x = x
    y = fht_butterfly(fht_butterfly(x))
    
    diff = (y - x).abs().max().item()
    assert diff < 1e-5, f"FHT not self-inverse: {diff:.2e}"
    print(f"  ✓ ||FHT(FHT(x)) - x||_max: {diff:.2e}")


def test_hadamard_layer():
    """Test HadamardLinear layer."""
    from src.model.fht_cuda import HadamardLinear
    
    print("Testing HadamardLinear...")
    layer = HadamardLinear(256, 128)
    x = torch.randn(2, 8, 256)
    y = layer(x)
    
    assert y.shape == (2, 8, 128), f"Wrong shape: {y.shape}"
    print(f"  ✓ Output shape: {y.shape}")
    
    # Test gradient flow
    y.sum().backward()
    assert layer.weight.grad is not None
    assert not layer.weight.grad.isnan().any()
    print(f"  ✓ Gradients flow correctly")


def test_tuple_wrapper():
    """Test HadamardTernaryLinearTuple for drop-in compatibility."""
    from src.model.fht_cuda import HadamardTernaryLinearTuple
    
    print("Testing HadamardTernaryLinearTuple...")
    layer = HadamardTernaryLinearTuple(256, 256)
    
    # Input: 32 parts of [B, T, in_o]
    x_parts = tuple(torch.randn(2, 8, 8) for _ in range(32))
    y_parts = layer(x_parts)
    
    assert len(y_parts) == 32, f"Wrong number of parts: {len(y_parts)}"
    assert y_parts[0].shape == (2, 8, 8), f"Wrong part shape: {y_parts[0].shape}"
    print(f"  ✓ Tuple interface works")


def test_model_instantiation():
    """Test both octonion and hadamard model variants."""
    from src.model.chassis import SpinNetConfig, SpinNet
    
    print("Testing model instantiation...")
    
    # Octonion (default)
    config_oct = SpinNetConfig(
        vocab_size=256, n_embd=128, n_layer=1, n_head=8, 
        block_size=32, algebra='octonion'
    )
    model_oct = SpinNet(config_oct)
    print(f"  ✓ Octonion model: {sum(p.numel() for p in model_oct.parameters()):,} params")
    
    # Hadamard
    config_had = SpinNetConfig(
        vocab_size=256, n_embd=128, n_layer=1, n_head=8,
        block_size=32, algebra='hadamard'
    )
    model_had = SpinNet(config_had)
    print(f"  ✓ Hadamard model: {sum(p.numel() for p in model_had.parameters()):,} params")


def test_forward_backward():
    """Test full forward and backward pass."""
    from src.model.chassis import SpinNetConfig, SpinNet
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing forward/backward on {device}...")
    
    config = SpinNetConfig(
        vocab_size=256, n_embd=256, n_layer=2, n_head=8,
        block_size=64, algebra='hadamard'
    )
    model = SpinNet(config).to(device)
    
    x = torch.randint(0, 256, (2, 16), device=device)
    targets = torch.randint(0, 256, (2, 16), device=device)
    
    logits, loss = model(x, targets)
    assert not loss.isnan(), "Loss is NaN"
    print(f"  ✓ Forward pass, loss: {loss.item():.4f}")
    
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is None:
            raise AssertionError(f"No gradient for {name}")
        if param.grad.isnan().any():
            raise AssertionError(f"NaN gradient for {name}")
    print(f"  ✓ Backward pass, gradients OK")


def main():
    print("=" * 50)
    print("SpinNet Clifford/Hadamard 32D Test Suite")
    print("=" * 50)
    print()
    
    tests = [
        test_fht_correctness,
        test_hadamard_orthonormality,
        test_fht_self_inverse,
        test_hadamard_layer,
        test_tuple_wrapper,
        test_model_instantiation,
        test_forward_backward,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
