import argparse
import cv2
import glob
import numpy as np
import os
import sys
import torch
import matplotlib
import OpenEXR
import Imath

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth_anything_v2.dpt import DepthAnythingV2


def save_exr(filename, depth):
    h, w = depth.shape
    header = OpenEXR.Header(w, h)
    header['channels'] = {
        'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }

    exr = OpenEXR.OutputFile(filename, header)
    exr.writePixels({'Z': depth.tobytes()})
    exr.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--exr', dest='exr', action='store_true', help='save float32 EXR')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Using device:', DEVICE)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    
    # Load from GitHub releases (no auth required)
    import urllib.request
    
    model_urls = {
        'vits': 'https://huggingface.co/lieryan/Depth-Anything-V2-Small/resolve/main/pytorch_model.bin',
        'vitb': 'https://huggingface.co/lieryan/Depth-Anything-V2-Base/resolve/main/pytorch_model.bin',
        'vitl': 'https://huggingface.co/lieryan/Depth-Anything-V2-Large/resolve/main/pytorch_model.bin',
        'vitg': 'https://huggingface.co/lieryan/Depth-Anything-V2-Giant/resolve/main/pytorch_model.bin'
    }
    
    cache_dir = os.path.expanduser('~/.cache/depth-anything-v2')
    os.makedirs(cache_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(cache_dir, f'depth_anything_v2_{args.encoder}.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f'Downloading {args.encoder} model... (~1-3 GB, may take a few minutes)')
        urllib.request.urlretrieve(model_urls[args.encoder], checkpoint_path)
        print('Download complete.')
    
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        if args.exr:
            save_exr(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.exr'), depth)
            exit(0)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)