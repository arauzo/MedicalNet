#!/usr/bin/env python3
"""
Script to inspect PyTorch model architecture and layer dimensions from a .pth file
"""

import torch
import argparse
from collections import OrderedDict


def format_size(num_params):
    """Format parameter count in human-readable format"""
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def print_model_info(model_path):
    """
    Load and print PyTorch model architecture and dimensions

    Args:
        model_path: Path to the .pth file
    """
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*80}\n")

    # Load the model file
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Handle different save formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Loaded from checkpoint dictionary (key: 'model')")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Loaded from checkpoint dictionary (key: 'state_dict')")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Loaded from checkpoint dictionary (key: 'model_state_dict')")
        else:
            # Assume it's a state_dict itself
            state_dict = checkpoint
            print("Loaded state dictionary")

        # Print additional checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_acc' in checkpoint or 'acc' in checkpoint:
            acc = checkpoint.get('best_acc', checkpoint.get('acc'))
            print(f"Accuracy: {acc}")
        print()
    elif isinstance(checkpoint, torch.nn.Module):
        print("Loaded full model object")
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint
        print("Loaded state dictionary")

    # Print architecture
    print(f"{'='*80}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*80}\n")

    print(f"{'Layer Name':<60} {'Shape':<25} {'Parameters':<15}")
    print(f"{'-'*60} {'-'*25} {'-'*15}")

    total_params = 0
    trainable_params = 0

    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params
            trainable_params += num_params  # Assuming all are trainable in state_dict

            shape_str = str(list(param.shape))
            print(f"{name:<60} {shape_str:<25} {format_size(num_params):<15}")

    print(f"{'-'*60} {'-'*25} {'-'*15}")
    print(f"{'TOTAL':<60} {'':<25} {format_size(total_params):<15}")
    print()

    # Summary statistics
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total parameters: {total_params:,} ({format_size(total_params)})")
    print(f"Total layers: {len(state_dict)}")

    # Calculate model size
    param_size = 0
    for param in state_dict.values():
        if isinstance(param, torch.Tensor):
            param_size += param.nelement() * param.element_size()

    buffer_size = 0
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"Model size: {size_mb:.2f} MB")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Inspect PyTorch model architecture and layer dimensions'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the .pth model file'
    )

    args = parser.parse_args()
    print_model_info(args.model_path)


if __name__ == '__main__':
    main()
