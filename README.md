# SurgX: Neuron–Concept Association for Explainable Surgical Phase Recognition (MICCAI 2025)

This repository contains the official implementation of the SurgX paper (MICCAI 2025):
**SurgX: Neuron–Concept Association for Explainable Surgical Phase Recognition**

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ailab-kyunghee.github.io/SurgX)

---

## Installation

Create a new conda environment and install all dependencies:

```bash
conda create -n SurgX --python=3.8
conda activate SurgX
pip install -r requirements.txt
```

---

## STEP 1: Neuron–Concept Annotation

Annotate neurons of **ASFormer**’s penultimate layer.

```bash
cd spr_models/ASFormer
```

There are four options for `--action`:

```bash
python main.py --action [train|extract_activations|predict|extract_contributions]
```

1. **Train the model:**

```bash
python main.py --action train
```

2. **Save activations of the train dataset:**

```bash
python main.py --action extract_activations
```

3. **Run the neuron–concept annotation pipeline:**

```bash
cd ../../
python 0_extract_sequence_features.py
python 0_extract_text_features.py
python 1_neuron_concept_annotation.py
```

**Optional – visualize which concepts neurons learn:**

```bash
python 2_visualize_neuron_concepts.py
python 3_make_videos.py
python 4_make_videos_with_info.py
python 5_make_integrated_videos_with_info.py
```

---

## STEP 2: Model Prediction Explanation

Generate explanations for **ASFormer** predictions.

```bash
cd spr_models/ASFormer
```

1. **Run prediction on the test dataset:**

```bash
python main.py --action predict
```

2. **Save contributions of the test dataset:**

```bash
python main.py --action extract_contributions
```

3. **Create explanations:**

```bash
cd ../../
python 6_explain_prediction.py
```

**Optional – visualize the explanations as MP4:**

```bash
python 7_make_mp4.py
```

---

## Concept Sets

Concept sets are located under the `concept_sets/` folder:

1. `CholecT45-W`
2. `CholecT45-S`
3. `ChoLec-270`

---

## Acknowledgments

The surgical phase recognition models are based on
[TeCNO](https://github.com/tobiascz/TeCNO) and
[ASFormer](https://github.com/ChinaYi/ASFormer).

The vision–language model used for neuron–concept annotation is
[SurgVLP](https://github.com/CAMMA-public/SurgVLP).

We thank all the authors for their efforts and open-source contributions.
