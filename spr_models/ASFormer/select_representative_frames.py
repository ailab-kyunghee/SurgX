import torch
import pickle
import os

# -------------------------
# Create output folder
# -------------------------
SAVE_DIR = "representative_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("./ASFormer_train_activations.pkl", "rb") as f:
    activations = pickle.load(f)

num_neurons = activations[0].shape[1]  # assume the same number of neurons for all videos

# -------------------------
# 1) Per neuron: global min/max → threshold-based selection
# -------------------------
results1 = {}
for neuron_idx in range(num_neurons):
    all_values = []
    for video_idx, tensor in enumerate(activations):
        values = tensor[0, neuron_idx, :]
        for frame_idx, val in enumerate(values):
            all_values.append((video_idx, frame_idx, val.item()))
    vals = [v[2] for v in all_values]
    max_val, min_val = max(vals), min(vals)
    thr = max_val - (max_val - min_val) * 0.003
    if max_val < 0.1:
        thr = 999999
    above = [(v, f, val) for v, f, val in all_values if val >= thr]
    results1[neuron_idx] = {"max": max_val, "min": min_val, "thr": thr, "frames": above}

# -------------------------
# 2) Per neuron: global Top-40 frames
# -------------------------
results2 = {}
for neuron_idx in range(num_neurons):
    all_values = []
    for video_idx, tensor in enumerate(activations):
        values = tensor[0, neuron_idx, :]
        for frame_idx, val in enumerate(values):
            all_values.append((video_idx, frame_idx, val.item()))
    top40 = sorted(all_values, key=lambda x: x[2], reverse=True)[:40]
    results2[neuron_idx] = top40

# -------------------------
# 3) Per neuron & per video: min/max → threshold-based selection
# -------------------------
results3 = {}
for neuron_idx in range(num_neurons):
    results3[neuron_idx] = {}
    for video_idx, tensor in enumerate(activations):
        values = tensor[0, neuron_idx, :].tolist()
        max_val, min_val = max(values), min(values)
        thr = max_val - (max_val - min_val) * 0.0001
        if max_val < 0.1:
            thr = 999999
        above = [(video_idx, i, val) for i, val in enumerate(values) if val >= thr]
        results3[neuron_idx][video_idx] = {"max": max_val, "min": min_val, "thr": thr, "frames": above}

# -------------------------
# 4) Per neuron & per video: Top-1 frame
# -------------------------
results4 = {}
for neuron_idx in range(num_neurons):
    results4[neuron_idx] = {}
    for video_idx, tensor in enumerate(activations):
        values = tensor[0, neuron_idx, :]
        max_val, frame_idx = torch.max(values, dim=0)
        results4[neuron_idx][video_idx] = (video_idx, frame_idx.item(), max_val.item())

# -------------------------
# Save TXT
# -------------------------
with open(os.path.join(SAVE_DIR, "1_Global_Threshold_0.003.txt"), "w") as f:
    for n, info in results1.items():
        f.write(f"Neuron {n} → max={info['max']:.4f}, min={info['min']:.4f}, thr={info['thr']:.4f}\n")
        for v, fr, val in info["frames"]:
            f.write(f"   Video {v}, Frame {fr}, Value {val:.4f}\n")
        f.write("-" * 50 + "\n")

with open(os.path.join(SAVE_DIR, "2_Global_Top40.txt"), "w") as f:
    for n, frames in results2.items():
        f.write(f"Neuron {n} → Top40 frames\n")
        for v, fr, val in frames:
            f.write(f"   Video {v}, Frame {fr}, Value {val:.4f}\n")
        f.write("-" * 50 + "\n")

with open(os.path.join(SAVE_DIR, "3_Video_wise_Threshold_0.0001.txt"), "w") as f:
    for n, vids in results3.items():
        f.write(f"Neuron {n}\n")
        for v, info in vids.items():
            f.write(f"  Video {v} → max={info['max']:.4f}, min={info['min']:.4f}, thr={info['thr']:.4f}\n")
            for vv, fr, val in info["frames"]:
                f.write(f"     Frame {fr}, Value {val:.4f}\n")
        f.write("-" * 50 + "\n")

with open(os.path.join(SAVE_DIR, "4_Video_wise_Top1.txt"), "w") as f:
    for n, vids in results4.items():
        f.write(f"Neuron {n}\n")
        for v, tup in vids.items():
            vid, fr, val = tup
            f.write(f"  Video {vid} → Frame {fr}, Value {val:.4f}\n")
        f.write("-" * 50 + "\n")

print("Saved four TXT result files.")

# -------------------------
# Save PKL (list per neuron)
# -------------------------
pkl1 = [[(v, f, val) for v, f, val in info["frames"]] for _, info in results1.items()]
with open(os.path.join(SAVE_DIR, "1_Global_Threshold_0.003.pkl"), "wb") as f:
    pickle.dump(pkl1, f)

pkl2 = [[(v, f, val) for v, f, val in frames] for _, frames in results2.items()]
with open(os.path.join(SAVE_DIR, "2_Global_Top40.pkl"), "wb") as f:
    pickle.dump(pkl2, f)

pkl3 = []
for _, vids in results3.items():
    neuron_list = []
    for v, info in vids.items():
        neuron_list.extend([(v, f, val) for v, f, val in info["frames"]])
    pkl3.append(neuron_list)
with open(os.path.join(SAVE_DIR, "3_Video_wise_Threshold_0.0001.pkl"), "wb") as f:
    pickle.dump(pkl3, f)

pkl4 = []
for _, vids in results4.items():
    neuron_list = []
    for v, tup in vids.items():
        neuron_list.append(tup)
    pkl4.append(neuron_list)
with open(os.path.join(SAVE_DIR, "4_Video_wise_Top1.pkl"), "wb") as f:
    pickle.dump(pkl4, f)

print(f"All TXT/PKL results saved under '{SAVE_DIR}'.")
