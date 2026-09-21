# MemeLens-VLM: Robust Multimodal Misogyny Classification with Qwen3-VL

Fine-tuning and robustness evaluation of **Qwen3-VL-8B-Instruct** for multimodal misogyny classification on the **MAMI (Multimedia Automatic Misogyny Identification)** dataset.

This project was developed for **Deep Learning (EE-559) at EPFL** by **Matteo Barberis, Giovanni Pomati, and Giorgio Panizzutti**.

## Overview

Multimodal content moderation systems can rely on both textual and visual cues, but their predictions may be sensitive to relatively simple modifications of the image.

This project studies whether a large vision-language model can be adapted efficiently to multimodal misogyny classification while remaining robust to synthetic visual obfuscations.

We fine-tuned **Qwen3-VL-8B-Instruct** using **Low-Rank Adaptation (LoRA)** and built an evaluation pipeline focused on three questions:

1. Can a large vision-language model be adapted efficiently to multimodal misogyny classification?
2. How much does performance deteriorate when the visual modality is partially obfuscated?
3. Can robustness-oriented fine-tuning generalize to **perturbation geometries not observed during training**?

## Model and Training

The base model is:

**Qwen3-VL-8B-Instruct**

Rather than fine-tuning the full model, we used **LoRA** for parameter-efficient adaptation.

Main tools:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT / LoRA
- scikit-learn
- NumPy
- pandas

The model was adapted to the MAMI image-text classification task while retaining the pretrained multimodal representations of Qwen3-VL.

## Dataset

Experiments were conducted on the **MAMI — Multimedia Automatic Misogyny Identification** dataset.

Each sample contains:

- an image;
- associated text;
- a misogyny label.

The task is well suited to multimodal robustness analysis because predictions may depend on either individual modality or on their interaction.

## Robustness Evaluation

The core experiment studies robustness to **synthetic visual obfuscations**.

Images are modified using sticker-like perturbations that partially obscure visual information while preserving the underlying semantic content.

The evaluation deliberately includes **previously unseen sticker geometries**, allowing us to test whether robustness learned during fine-tuning transfers beyond the exact augmentation patterns used during training.

Two evaluation settings are considered:

- **Image only:** measures sensitivity to corruption of the visual modality.
- **Image + text:** evaluates whether textual information can compensate for degraded visual information.

Performance is measured using **macro-F1**, with particular attention to the degradation between clean and perturbed inputs.

## Results

Robustness-oriented fine-tuning substantially reduced the macro-F1 degradation caused by previously unseen visual perturbations.

| Evaluation setting | Macro-F1 degradation before | Macro-F1 degradation after |
| ------------------ | --------------------------: | -------------------------: |
| Image only         |                        5.9% |                   **1.5%** |
| Image + text       |                        7.0% |                   **1.1%** |

The strongest improvement was observed in the multimodal image-text setting, where the degradation decreased from **7.0% to 1.1%**.

Prediction files and experiment summaries are stored under `Hate_Project/results/`, allowing the clean, obfuscated, and fine-tuned conditions to be inspected separately.

Overall, the experiments indicate that robustness improvements can transfer beyond the precise perturbation geometry used during training.

## Experimental Pipeline

```text
MAMI image + text samples
          |
          v
   Data preprocessing
          |
          v
 Qwen3-VL-8B-Instruct
          |
          v
     LoRA fine-tuning
          |
          +----------------------+
          |                      |
          v                      v
     Clean evaluation     Visual perturbation
                                 |
                                 v
                     Unseen sticker geometries
                                 |
                                 v
                       Robustness evaluation
                                 |
                                 v
                         Macro-F1 analysis
```

## Collaboration

This was a **three-person project developed collaboratively by Matteo Barberis, Giovanni Pomati, and Giorgio Panizzutti**.

Model development, experimentation, evaluation, and analysis were carried out jointly throughout the project rather than being divided into strictly separate individual components.

---

**EPFL EE-559 — Deep Learning**
