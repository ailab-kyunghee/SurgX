---
layout: project_page
title: "[MICCAI 2025] SurgX: Neuron-Concept Association for Explainable Surgical Phase Recognition"
authors:
  - Ka Young Kim, Hyeon Bae Kim, Seong Tae Kim†
affiliations:
  - Kyung Hee University, Yongin, Republic of Korea
paper: https://arxiv.org/pdf/2507.15418v1
code: https://github.com/ailab-kyunghee/SurgX
---

<!-- Abstract -->
<div class="columns is-centered has-text-centered">
  <div class="column is-four-fifths">
    <h2>Abstract</h2>
    <div class="content has-text-justified">
      Surgical phase recognition plays a crucial role in surgical workflow analysis, enabling applications such as monitoring, skill assessment, and workflow optimization. However, deep learning models remain black-boxes, limiting interpretability and trust. 
      <b>SurgX</b> is a novel concept-based explanation framework that associates neurons with human-interpretable surgical concepts. We construct concept sets tailored to cholecystectomy, select representative neuron activation sequences, and annotate neurons with concepts. 
      By evaluating on TeCNO and Causal ASFormer using Cholec80, we demonstrate that SurgX provides meaningful explanations and improves transparency in surgical AI.  
      <br><br>
      The code is available at <a href="https://github.com/ailab-kyunghee/SurgX" target="_blank">GitHub</a>.
    </div>
  </div>
</div>

---

## 🚀 Motivation
Deep models for **surgical phase recognition** are highly accurate but opaque.  
SurgX addresses two key challenges:
1. Lack of interpretability in temporal models.  
2. Difficulty in associating neurons with domain-specific surgical knowledge.

---

## 🧩 Methodology
SurgX consists of three core components:
1. **Concept Set Construction**  
   - CholecT45-W (30 keywords)  
   - CholecT45-S (100 sentences)  
   - ChoLec-270 (270 domain-specific surgical concepts)  
2. **Neuron Representative Sequence Selection**  
   - Extract highly-activated frames → build sequences (dilated / contiguous).  
3. **Neuron-Concept Annotation**  
   - Use vision-language embeddings (SurgVLP) to match neuron activations with textual concepts.  

![Method Overview](/static/image/surgx_method.png)  
*Figure 1: Neuron–concept association pipeline in SurgX.*

---

## 📊 Results

### Quantitative Evaluation
We compare concept set construction, frame selection, and sequence design.

**Table 1: Concept Set Analysis**  
![Table 1](/static/image/table1.png)

**Table 2: Ablation on Frame Selection**  
![Table 2](/static/image/table2.png)

---

## 🔍 Qualitative Analysis
SurgX enables explaining both correct and incorrect predictions:

- **Correct case:** model uses concepts like *“insert a port”* or *“hepatocystic triangle”* from previous frames.  
- **Failure case:** neurons annotated with *“cystic artery is isolated between clips”* often mislead predictions.  

![Qualitative Results](/static/image/surgx_qualitative.png)  
*Figure 2: Explanations for TeCNO and Causal ASFormer predictions.*

---

## ✨ Contributions
- First **concept-based explanation framework** for surgical phase recognition.  
- Novel **concept set construction** tailored to cholecystectomy.  
- Extensive evaluation on **TeCNO** and **Causal ASFormer**.  
- Improves transparency and reliability of surgical AI models:contentReference[oaicite:1]{index=1}.

---
