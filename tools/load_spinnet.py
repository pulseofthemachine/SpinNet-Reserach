"""
SpinNet Loader - Load .spinnet files back into PyTorch
-------------------------------------------------------
Loads compressed .spinnet models for inference/verification in PyTorch.

Usage:
    python tools/load_spinnet.py experiments/out-tinystories-octonion/model.spinnet --prompt "Once upon a time"
"""

import argparse
import json
import struct
import numpy as np
import torch
import tiktoken

def read_array(f) -> np.ndarray:
    """Read numpy array with header: dtype_code(1) + ndim(1) + shape(4*ndim) + data"""
    dtype_map = {0: np.int8, 1: np.int32, 2: np.float16, 3: np.float32}
    dtype_code = struct.unpack('B', f.read(1))[0]
    ndim = struct.unpack('B', f.read(1))[0]
    shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
    dtype = dtype_map[dtype_code]
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    data = np.frombuffer(f.read(size), dtype=dtype).reshape(shape)
    return data.copy()


def unpack_bitmask_ternary(bitmask: np.ndarray, sign_bits: np.ndarray, 
                           shape: tuple, num_nonzero: int) -> torch.Tensor:
    """
    Reconstruct ternary weights from bitmask + sign representation.
    
    bitmask: 1 bit per position (1 = non-zero)
    sign_bits: 1 bit per non-zero (0 = -1, 1 = +1)
    """
    total = int(np.prod(shape))
    
    # Expand bitmask to boolean array
    positions = np.zeros(total, dtype=np.float32)
    
    # Unpack bitmask
    nonzero_indices = []
    for byte_idx, byte_val in enumerate(bitmask):
        for bit in range(8):
            pos = byte_idx * 8 + bit
            if pos >= total:
                break
            if byte_val & (1 << bit):
                nonzero_indices.append(pos)
    
    # Unpack signs
    signs = []
    for byte_idx in range(len(sign_bits)):
        byte_val = sign_bits[byte_idx]
        for bit in range(8):
            idx = byte_idx * 8 + bit
            if idx >= num_nonzero:
                break
            sign = 1.0 if (byte_val & (1 << bit)) else -1.0
            signs.append(sign)
    
    # Reconstruct
    for i, pos in enumerate(nonzero_indices[:num_nonzero]):
        positions[pos] = signs[i]
    
    return torch.tensor(positions.reshape(shape), dtype=torch.float32)


def load_spinnet(path: str, device='cpu') -> dict:
    """
    Load a .spinnet file and return state_dict compatible with SpinNet model.
    """
    print(f"Loading SpinNet model from {path}")
    
    with open(path, 'rb') as f:
        # Magic number
        magic = f.read(4)
        assert magic == b'SPIN', f"Invalid magic number: {magic}"
        
        # Version
        version = struct.unpack('<H', f.read(2))[0]
        print(f"  Format version: {version}")
        assert version == 4, f"Unsupported version: {version} (expected 4)"
        
        # Config JSON
        config_len = struct.unpack('<I', f.read(4))[0]
        config_json = f.read(config_len).decode('utf-8')
        config = json.loads(config_json)
        print(f"  Config: {config}")
        
        state_dict = {}
        
        while True:
            marker = f.read(1)
            if not marker:
                break
            
            # Check for END marker
            if marker == b'E':
                peek = f.read(3)
                if peek == b'ND!':
                    break
                else:
                    # It's an Embedding marker, not END
                    f.seek(-3, 1)  # Go back
            
            # Read name
            name_len = struct.unpack('<H', f.read(2))[0]
            name = f.read(name_len).decode('utf-8')
            
            if marker == b'E':
                # Embedding - INT8 + scale
                scale = struct.unpack('<f', f.read(4))[0]
                data = read_array(f)
                # Dequantize
                tensor = torch.tensor(data.astype(np.float32) * scale)
                state_dict[name] = tensor
                print(f"  [EMBED] {name}: {tensor.shape}")
                
            elif marker == b'B':
                # Bitmask ternary weight
                ndim = struct.unpack('<B', f.read(1))[0]
                shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
                num_nonzero = struct.unpack('<I', f.read(4))[0]
                
                bitmask_len = struct.unpack('<I', f.read(4))[0]
                bitmask = np.frombuffer(f.read(bitmask_len), dtype=np.uint8)
                
                sign_len = struct.unpack('<I', f.read(4))[0]
                sign_bits = np.frombuffer(f.read(sign_len), dtype=np.uint8)
                
                tensor = unpack_bitmask_ternary(bitmask, sign_bits, shape, num_nonzero)
                state_dict[name] = tensor
                print(f"  [WEIGHT] {name}: {tensor.shape}")
                
            elif marker == b'S':
                # Scale (beta) - FP16
                data = read_array(f)
                tensor = torch.tensor(data.astype(np.float32))
                state_dict[name] = tensor
                print(f"  [SCALE] {name}: {tensor.shape}")
                
            elif marker == b'N':
                # Norm weight - FP16
                data = read_array(f)
                tensor = torch.tensor(data.astype(np.float32))
                state_dict[name] = tensor
                print(f"  [NORM] {name}: {tensor.shape}")
                
            elif marker == b'R':
                # RoPE - FP32 complex
                data = read_array(f)
                # data is [seq_len, dim/2, 2] (real/imag pairs)
                tensor = torch.view_as_complex(torch.tensor(data))
                state_dict[name] = tensor
                print(f"  [ROPE] {name}: {tensor.shape}")
            else:
                print(f"  [UNKNOWN] marker={marker}")
                break
    
    return state_dict, config


def sample_from_spinnet(spinnet_path: str, prompt: str, max_tokens: int = 100,
                        temperature: float = 0.8, device: str = 'cuda'):
    """Load a .spinnet and generate text."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import SpinNetConfig, SpinNet
    
    # Load weights
    state_dict, config = load_spinnet(spinnet_path, device)
    
    # Create model
    model_config = SpinNetConfig(**config)
    model = SpinNet(model_config)
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    
    # Tokenize
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, device=device)[None, ...]
    
    # Generate
    print(f"\nGenerating from: '{prompt}'")
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            y = model.generate(x, max_tokens, temperature=temperature, top_k=50)
    
    output = enc.decode(y[0].tolist())
    print(f"\nOutput:\n{output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and sample from .spinnet model")
    parser.add_argument("spinnet", help="Path to .spinnet file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    sample_from_spinnet(args.spinnet, args.prompt, args.max_tokens, 
                        args.temperature, args.device)
