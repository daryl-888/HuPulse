import argparse
import os
import sys
from typing import Dict, Any

import torch

# Allow running from project root via train_job.sh
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from Model_Def.MS4 import S4Model, MS4_1D  # noqa: F401
except Exception as e:  # Fallback if relative import path differs
    print(f"[WARN] Could not import models from Model_Def: {e}")
    raise

CKPT_PATH_DEFAULT = "/home/yshen28/PulseDB_multi/091857/checkpoint_epoch_38.pth"


def strip_prefix_in_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def attempt_load(model: torch.nn.Module, raw_obj: Any) -> bool:
    """Try multiple strategies to extract a usable state_dict from raw_obj and load it into model.
    Returns True if successful (even with missing/unexpected keys), else False.
    """
    candidates = []
    # Case 1: raw object is already a state dict
    if isinstance(raw_obj, dict):
        if raw_obj and all(isinstance(v, torch.Tensor) for v in raw_obj.values()):
            candidates.append(("<root>", raw_obj))
        # Common wrappers
        for k in ["state_dict", "model_state_dict", "model", "net", "network"]:
            v = raw_obj.get(k)
            if isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
                candidates.append((k, v))

    # If entire model object was saved
    if not candidates and hasattr(raw_obj, "state_dict"):
        sd = raw_obj.state_dict()
        candidates.append(("object.state_dict()", sd))

    tried = 0
    for name, sd in candidates:
        tried += 1
        # Try various prefix stripping patterns
        variant_keys = [sd]
        prefixes = ["module.", "model.", "net.", "network."]
        # Compose single-level strips
        for p in prefixes:
            variant_keys.append(strip_prefix_in_state_dict(sd, p))
        # Compose double strips (e.g. module.model.)
        for p1 in prefixes:
            for p2 in prefixes:
                variant_keys.append(strip_prefix_in_state_dict(strip_prefix_in_state_dict(sd, p1), p2))

        for idx, cand in enumerate(variant_keys):
            try:
                missing, unexpected = model.load_state_dict(cand, strict=False)
                total_params = len(cand)
                matched = total_params - len(missing) - len(unexpected)
                match_ratio = matched / max(1, total_params)
                print(f"[INFO] Attempt '{name}' variant {idx}: matched {matched}/{total_params} ({match_ratio:.1%}), missing={len(missing)}, unexpected={len(unexpected)}")
                # Consider success if at least half match OR no unexpected keys (typical partial fine-tune)
                if match_ratio >= 0.5 or (not unexpected and match_ratio > 0):
                    if missing:
                        print(f"[WARN] Missing keys (first 10): {missing[:10]}")
                    if unexpected:
                        print(f"[WARN] Unexpected keys (first 10): {unexpected[:10]}")
                    return True
            except Exception as e:
                # Skip failed variant
                print(f"[DEBUG] Failed variant {idx} for candidate '{name}': {e}")
                continue
    print("[ERROR] Could not load any candidate state dict into model.")
    return False


def build_model(args) -> torch.nn.Module:
    if args.arch == "s4model":
        model = S4Model(
            d_input=args.d_input,
            d_output=args.d_output,
            d_state=args.d_state,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            l_max=args.l_max,
            bidirectional=not args.no_bidirectional,
            pooling=not args.no_pooling,
        )
    elif args.arch == "ms4_1d":
        model = MS4_1D(
            #num_static_features=args.num_static_features,
            num_static_features=3,
            #num_BP=args.d_output if args.d_output is not None else 1,
            num_BP=1,
            #static_hidden=args.static_hidden,
            #fusion_hidden=args.fusion_hidden,
            #s4_d_input=args.d_input,
            #s4_d_model=args.d_model,
            #s4_n_layers=args.n_layers,
            #s4_pooling=not args.no_pooling,
            #s4_l_max=args.l_max,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Load S4 checkpoint and report status")
    p.add_argument("--ckpt", type=str, default=CKPT_PATH_DEFAULT, help="Path to checkpoint .pth file")
    p.add_argument("--arch", type=str, default="ms4_1d", choices=["s4model", "ms4_1d"], help="Model architecture to instantiate")
    #p.add_argument("--arch", type=str, default="s4model", choices=["s4model", "ms4_1d"], help="Model architecture to instantiate")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Model hyperparameters
    p.add_argument("--d_input", type=int, default=1)
    p.add_argument("--d_output", type=int, default=1)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--l_max", type=int, default=1024)
    p.add_argument("--no_bidirectional", action="store_true")
    p.add_argument("--no_pooling", action="store_true")
    # MS4_1D specific
    p.add_argument("--num_static_features", type=int, default=3)
    p.add_argument("--static_hidden", type=int, default=16)
    p.add_argument("--fusion_hidden", type=int, default=32)
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("========== S4 Checkpoint Loading Test ==========")
    print(f"Checkpoint path: {args.ckpt}")
    print(f"Device: {device}")
    if not os.path.isfile(args.ckpt):
        print(f"[ERROR] Checkpoint file not found: {args.ckpt}")
        return

    print("[INFO] Building model...")
    model = build_model(args).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model architecture: {args.arch}")
    print(f"[INFO] Total parameters: {total_params:,}")

    print("[INFO] Loading checkpoint raw object...")
    raw = torch.load(args.ckpt, map_location=device)
    print(f"[INFO] Raw checkpoint type: {type(raw)}")
    if isinstance(raw, dict):
        print(f"[INFO] Top-level keys: {list(raw.keys())[:10]}")

    success = attempt_load(model, raw)
    if success:
        print("[SUCCESS] Checkpoint (partially) loaded into model.")
    else:
        print("[FAIL] Failed to load checkpoint.")

    # Simple forward pass test (optional) if model expects an input
    try:
        with torch.no_grad():
            if args.arch == "s4model":
                dummy = torch.randn(2, args.d_input, args.l_max, device=device)
                out = model(dummy)
                print(f"[INFO] Forward pass output shape: {tuple(out.shape)}")
            elif args.arch == "ms4_1d":
                dummy_ppg = torch.randn(2, args.d_input, args.l_max, device=device)
                dummy_static = torch.randn(2, args.num_static_features, device=device)
                out = model(dummy_ppg, dummy_static)
                print(f"[INFO] Forward pass output shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"[WARN] Forward pass test skipped due to error: {e}")

    print("========== Done ==========")


if __name__ == "__main__":
    main()
