"""
Explore PulseDB data structure on Carya.
Run with: python scripts/explore_data.py --data_root /project/rhu/PulseBP
"""
import os
import sys
import argparse
import numpy as np

def try_load_mat73(path):
    try:
        import mat73
        data = mat73.loadmat(path)
        return data, 'mat73'
    except Exception as e:
        return None, str(e)

def try_load_scipy(path):
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        return data, 'scipy'
    except Exception as e:
        return None, str(e)

def try_load_h5py(path):
    try:
        import h5py
        data = {}
        with h5py.File(path, 'r') as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
            f.visititems(visitor)
        return data, 'h5py'
    except Exception as e:
        return None, str(e)

def summarize_dict(d, prefix='', depth=0):
    if depth > 3:
        return
    for k, v in d.items():
        if k.startswith('__'):
            continue
        if isinstance(v, np.ndarray):
            print(f"{'  '*depth}{prefix}{k}: ndarray shape={v.shape} dtype={v.dtype} "
                  f"min={v.min():.3f} max={v.max():.3f} mean={v.mean():.3f}")
        elif isinstance(v, dict):
            print(f"{'  '*depth}{prefix}{k}: dict with {len(v)} keys")
            summarize_dict(v, depth=depth+1)
        elif isinstance(v, (list, tuple)):
            print(f"{'  '*depth}{prefix}{k}: {type(v).__name__} len={len(v)}")
        else:
            print(f"{'  '*depth}{prefix}{k}: {type(v).__name__} = {v}")

def explore_file(path):
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Size: {os.path.getsize(path) / 1e6:.1f} MB")

    data, method = try_load_mat73(path)
    if data is None:
        data, method = try_load_scipy(path)
    if data is None:
        data, method = try_load_h5py(path)
    if data is None:
        print(f"Could not load file: {method}")
        return

    print(f"Loaded with: {method}")
    print(f"Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    if isinstance(data, dict):
        summarize_dict(data)

def explore_dir(path):
    print(f"\n{'='*60}")
    print(f"Directory: {path}")

    all_files = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, path)
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  {rel:60s} {size_mb:8.1f} MB")
            all_files.append(fpath)

    return all_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/project/rhu/PulseBP')
    args = parser.parse_args()

    print(f"Exploring: {args.data_root}")

    # Top-level listing
    for entry in sorted(os.listdir(args.data_root)):
        fpath = os.path.join(args.data_root, entry)
        if os.path.isdir(fpath):
            print(f"\n[DIR] {entry}/")
            all_files = explore_dir(fpath)
            # Load and summarize the first .mat or .h5 file found
            for f in all_files[:2]:
                if f.endswith(('.mat', '.h5', '.hdf5')):
                    explore_file(f)
                    break
        elif os.path.isfile(fpath):
            ext = os.path.splitext(fpath)[1]
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"\n[FILE] {entry} ({size_mb:.1f} MB)")
            if ext in ('.mat', '.h5', '.hdf5', '.npz'):
                explore_file(fpath)

if __name__ == '__main__':
    main()
