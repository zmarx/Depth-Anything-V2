#!/usr/bin/env python3
"""Download Depth-Anything-V2 models to local cache."""

import os
import sys
import urllib.request
import argparse

def download_model(encoder='vitl'):
    """Download the specified model."""
    
    model_urls = {
        'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true',
        'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true',
        'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true',
        'vitg': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth?download=true'
    }
    
    if encoder not in model_urls:
        print(f"Unknown encoder: {encoder}")
        print(f"Available: {', '.join(model_urls.keys())}")
        return False
    
    cache_dir = os.path.expanduser('./checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(cache_dir, f'depth_anything_v2_{encoder}.pth')
    
    if os.path.exists(checkpoint_path):
        size_gb = os.path.getsize(checkpoint_path) / (1024**3)
        print(f"✓ Model already cached: {checkpoint_path}")
        print(f"  Size: {size_gb:.2f} GB")
        return True
    
    print(f"Downloading {encoder} model...")
    print(f"URL: {model_urls[encoder]}")
    print(f"Destination: {checkpoint_path}")
    print()
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            total_mb = total_size / (1024**2)
            downloaded_mb = downloaded / (1024**2)
            percent = min(100, int((downloaded / total_size) * 100))
            
            bar_length = 40
            filled = int(bar_length * downloaded / total_size)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            sys.stdout.write(f'\r[{bar}] {percent:3d}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)')
            sys.stdout.flush()
        
        urllib.request.urlretrieve(model_urls[encoder], checkpoint_path, show_progress)
        print()
        print()
        
        size_gb = os.path.getsize(checkpoint_path) / (1024**3)
        print(f"✓ Download complete!")
        print(f"  Cached at: {checkpoint_path}")
        print(f"  Size: {size_gb:.2f} GB")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Depth-Anything-V2 models')
    parser.add_argument('--encoder', type=str, default='vitl', 
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to download')
    args = parser.parse_args()
    
    success = download_model(args.encoder)
    sys.exit(0 if success else 1)
