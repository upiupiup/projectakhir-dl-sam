# SAM + Adaptive Optimizers: Exploring Sharpness-Aware Minimization with AdamW and Ranger

This repository contains the implementation and experiments for exploring the combination of **Sharpness-Aware Minimization (SAM)** (Foret et al., ICLR 2021) with **adaptive optimizers** such as **AdamW** and **Ranger**, tested on **CIFAR-10** using **ResNet-18**.  
The goal of this project is to analyze how SAM behaves when used with optimizers beyond SGD, focusing on convergence, generalization, and training stability.

## 1. Background

The original SAM paper proposes an optimization technique that improves generalization by seeking parameters that lie in **flat minima**—regions in the loss landscape where small perturbations of weights lead to minimal changes in loss.

Formally, SAM minimizes the following objective:

\[
\min_w \max_{\|\varepsilon\|_2 \le \rho} L(w + \varepsilon)
\]

This ensures that the chosen parameters are robust to perturbations and generalize better.  
In the original implementation, SAM was applied on **SGD with momentum**.  
This project explores how SAM performs when combined with **adaptive optimizers** like **AdamW** and **Ranger**, which are widely used in modern deep learning workflows.

## 2. Research Questions

1. Does SAM still improve generalization when combined with adaptive optimizers such as AdamW or Ranger?
2. How does the convergence speed of SAM(AdamW) compare with standard AdamW and SAM(SGD)?
3. What are the trade-offs between stability and computational cost when applying SAM on adaptive optimizers?

## 3. Experimental Setup

| Component | Description |
|------------|-------------|
| **Dataset** | CIFAR-10 (50k train, 10k test, 10 classes, 32×32 RGB) |
| **Model** | ResNet-18 |
| **Optimizers** | SGD, SAM(SGD), AdamW, SAM(AdamW), (optional) Ranger, SAM(Ranger) |
| **Epochs** | 100–150 |
| **Batch size** | 128 |
| **Learning rate** | SGD: 0.1, AdamW: 1e-3 (cosine or step schedule) |
| **SAM radius (ρ)** | 0.02 and 0.05 |
| **Hardware** | 1× GPU (Colab T4 / RTX 3060) |

## 4. Repository Structure

```

sam-adaptive-optimizers/
├─ src/
│   ├─ models.py             # ResNet-18 and other architectures
│   ├─ sam_optimizer.py      # SAM and ASAM implementation
│   ├─ train.py              # Training and evaluation loop
│   ├─ utils.py              # Metrics, checkpointing, plotting
│
├─ experiments/
│   ├─ configs/              # YAML/JSON config files for each experiment
│   └─ logs/                 # Training logs and results
│
├─ reports/
│   ├─ figures/              # Curves, comparison charts, loss landscapes
│   └─ tables/               # Summaries of results
│
├─ requirements.txt
└─ README.md

````

## 5. Implementation Notes
- SAM is implemented as a **wrapper** around the base optimizer:
  ```python
  base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, adaptive=True)
* For SGD:

  ```python
  base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
  optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
  ```
* The `adaptive=True` flag corresponds to **ASAM** (Adaptive SAM), where the perturbation magnitude scales with parameter norms.

## 6. Expected Results

| Optimizer   | Accuracy (Expected Trend) | Notes                                      |
| ----------- | ------------------------- | ------------------------------------------ |
| SGD         | baseline                  | slower but stable                          |
| SAM(SGD)    | +1–2%                     | more stable, better generalization         |
| AdamW       | faster convergence        | slightly less generalization               |
| SAM(AdamW)  | moderate gain             | smoother loss curve, slower but consistent |
| Ranger      | adaptive fast             | may overfit                                |
| SAM(Ranger) | balanced                  | potentially strong stability               |

## 7. References
* Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur.
  *Sharpness-Aware Minimization for Efficiently Improving Generalization.*
  ICLR 2021. [Paper PDF](https://arxiv.org/abs/2010.01412) | [Official GitHub (Google Research)](https://github.com/google-research/sam)
* Leslie Smith et al., *AdamW and Super-Convergence*, ICLR 2018.
* Wright, *Ranger Optimizer (RAdam + Lookahead)*, GitHub 2019.

## 8. Acknowledgement
This repository is inspired by the original SAM implementation by [Google Research](https://github.com/google-research/sam) and minimal PyTorch adaptations by the open-source community.
Developed as part of the final project for undergraduate coursework in Machine Learning, Faculty of Computer Science, Universitas Brawijaya (2025).

## 9. Contributors

* **Aufii Fathin Nabila** – Implementation, experiment design, paper writing
* **Dwi Cahya Maulani** – Model setup, analysis, visualization
