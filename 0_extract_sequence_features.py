import os
import glob
import pickle
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from mmengine.config import Config
import SurgVLP.surgvlp as surgvlp

# =========================
# Load PeskaVLP image encoder
# =========================
def load_peskavlp_image_encoder(config_path: str, device: str = None):
    """
    Args:
        config_path: e.g., './config_peskavlp.py'
    Returns:
        (model, preprocess_transform, feature_dim, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = Config.fromfile(config_path, lazy_import=False)['config']
    model, _ = surgvlp.load(configs.model_config)
    model = model.to(device)
    model.eval()

    # Preprocessing: ImageNet normalization (mirrors zero-shot code's unnormalize counterpart)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Image embedding dimension is typically 768 or 1024 depending on config.
    # We assume 768 here for simplicity.
    feature_dim = 768
    return model, preprocess, feature_dim, device

# =========================
# Path / preprocessing / encoding utils
# =========================
def tuple_to_paths(vid: int, frame_1fps: int) -> List[str]:
    """
    Convert 1fps frame index to 25fps (base = f*25+1), then gather up to 10 frames
    spaced by 5 seconds (=125 frames) backward including the current frame.
    Only existing files are returned.

    Args:
        vid: zero-based video index
        frame_1fps: frame index at 1 fps
    Returns:
        List of image paths (most recent first)
    """
    vid += 1
    base = frame_1fps * 25 + 1
    paths = []
    for i in range(10):
        fnum = base - i * 125
        if fnum <= 0:
            break
        p = f"/data2/local_datasets/cholec80/cholec_split_360x640_1fps/video{vid:02d}/video{vid:02d}_{fnum:06d}.png"
        if os.path.exists(p):
            paths.append(p)
    return paths

def load_images_as_batch(paths: List[str], preprocess, device: str) -> torch.Tensor:
    """
    Load multiple images and return a (N, 3, 224, 224) tensor.

    Args:
        paths: list of image paths
        preprocess: torchvision transform
        device: torch device string
    Returns:
        Tensor of shape (N, 3, 224, 224) on device, or empty tensor if no paths
    """
    batch = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        batch.append(preprocess(img))
    if len(batch) == 0:
        return torch.empty(0, 3, 224, 224, device=device)
    batch = torch.stack(batch, dim=0).to(device)
    return batch

@torch.no_grad()
def peskavlp_encode_images(model, images: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of images using PeskaVLP (L2-normalized embeddings).

    Args:
        images: (N, 3, 224, 224)
    Returns:
        (N, D) L2-normalized image embeddings
    """
    if images.numel() == 0:
        return images.new_zeros((0, 768))
    # Call signature aligned with zero-shot code
    out = model(images, None, mode='video')
    img_emb = out['img_emb']  # (N, D)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb

@torch.no_grad()
def extract_tuple_feature(model, preprocess, device: str, vid: int, frame_1fps: int, feature_dim: int) -> torch.Tensor:
    """
    For a single (vid, frame_1fps) tuple, average features over up to 10 frames
    (only existing files). Returns a (D,) tensor.

    Args:
        model: PeskaVLP model
        preprocess: torchvision transform
        device: torch device string
        vid: zero-based video index
        frame_1fps: frame index at 1 fps
        feature_dim: embedding dimension
    Returns:
        Tensor of shape (D,)
    """
    paths = tuple_to_paths(vid, frame_1fps)
    if not paths:
        return torch.zeros(feature_dim, device=device)
    images = load_images_as_batch(paths, preprocess, device)  # (N, 3, 224, 224)
    feats = peskavlp_encode_images(model, images)             # (N, D)
    return feats.mean(dim=0)                                  # (D,)

# =========================
# Main processing routine
# =========================
def process_one_pkl(input_pkl: str, output_dir: str, model, preprocess, device: str, feature_dim: int):
    """
    For a single input PKL:
      - Input format: [ [ (vid, frame, val), ... ], [ ... ], ... ]  (list per neuron)
      - Extract features for each tuple and stack per neuron -> (num_tuples, D)
      - Save as a list of tensors (one per neuron) back to PKL
    """
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)  # list of neurons; each neuron is a list of tuples

    per_neuron_tensors = []  # each element: (num_tuples, D)

    for neuron in tqdm(data, desc=f"[{os.path.basename(input_pkl)}] Neurons", position=0, leave=True):
        tuple_feat_list = []
        for tup in tqdm(neuron, desc="  Tuples", position=1, leave=False):
            if not isinstance(tup, (list, tuple)) or len(tup) < 2:
                continue
            vid, frame_1fps = int(tup[0]), int(tup[1])
            feat = extract_tuple_feature(model, preprocess, device, vid, frame_1fps, feature_dim)  # (D,)
            tuple_feat_list.append(feat)

        if len(tuple_feat_list) == 0:
            neuron_tensor = torch.empty(0, feature_dim)
        else:
            neuron_tensor = torch.stack(tuple_feat_list, dim=0)  # (num_tuples, D)

        per_neuron_tensors.append(neuron_tensor.cpu())

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(input_pkl))
    for p in per_neuron_tensors:
        print(p.shape)
    with open(out_path, "wb") as f:
        pickle.dump(per_neuron_tensors, f)

    print(f"Saved -> {out_path}")

def main():
    input_dir  = "spr_models/ASFormer/representative_frames"
    output_dir = "extracted_features/sequence"
    config_path = "SurgVLP/tests/config_peskavlp.py"   # same config file used in zero-shot code

    model, preprocess, feature_dim, device = load_peskavlp_image_encoder(config_path)
    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))

    for idx, pkl_path in enumerate(pkl_files, start=1):
        print(f"\n=== [{idx}/{len(pkl_files)}] Processing {os.path.basename(pkl_path)} ===")
        process_one_pkl(pkl_path, output_dir, model, preprocess, device, feature_dim)

if __name__ == "__main__":
    main()
