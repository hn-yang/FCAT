# Facial Core Anchoring Triangle: Enhancing Micro-Expression Spotting through Geometric Alignment.

This repository provides the implementation of Facial Core Anchoring Triangle (FCAT): Enhancing Micro-Expression Spotting through Geometric Alignment, as described in our manuscript submitted to *The Visual Computer*.

> **Manuscript note:** This code is directly related to the manuscript currently submitted to *The Visual Computer*.  
> **Citation request:** If you use this code, please cite our manuscript.

---

## 1. Repository Contents

- `FCATtraincas.py` — training/evaluation script for **CAS(ME)<sup>2</sup>**
- `FCATtrain_samm.py` — training/evaluation script for **SAMM-LV**
- `FCATcas_util.py` — utilities for CAS(ME)<sup>2</sup>
- `FCATsamm_util.py` — utilities for SAMM-LV
- `requirements.txt` — required Python packages

---

## 2. Requirements & Installation

### 2.1 Python environment
- Python **3.9** (recommended: 3.8+)

Install dependencies:

```bash
pip install -r requirements.txt

````
---

## 3. Datasets

We evaluate Facial Core Anchoring Triangle: Enhancing Micro-Expression Spotting through Geometric Alignment on:

- **CAS(ME)<sup>2</sup>**
- **SAMM-LV**

**Data availability:** Due to copyright issues, the **SAMM-LV** and **CAS(ME)<sup>2</sup>** datasets used in this study cannot be provided directly. These datasets can be accessed through the following links:

- CAS(ME)<sup>2</sup> dataset: http://casme.psych.ac.cn/casme/c3  
- SAMM-LV dataset: https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php  

Please obtain the datasets through the official channels and follow their terms of use.

---

## 4. Usage

### 4.1 Run on CAS(ME)<sup>2</sup>

```bash
python FCATtraincas.py
````
### 4.2 Run on SAMM-LV
```bash
python FCATtrain_samm.py
````
If the scripts require dataset paths, please modify the dataset root path variables in `FCATtraincas.py` and `FCATtrain_samm.py` according to your local directory structure.


---

## 5. Output & Evaluation

The scripts report spotting performance (e.g., **F1-score**) in the terminal output and/or save results to local files depending on the script settings.

Reported results in our manuscript:

- CAS(ME)<sup>2</sup>: **F1-score = 0.4405**
- SAMM-LV: **F1-score = 0.3381**

(Results may vary slightly due to randomness and environment differences.)

---

## 6. Notes on Key Components

FCAT includes three steps:

1. **Face detection and alignment:** Detect facial key points per frame and align faces using a **facial core anchoring triangle** built from three stable points, then crop the facial region.

2. **Feature extraction:** Select **13 ROIs** and extract **optical flow** features (main/secondary directions) to form temporal motion curves.

3. **Expression spotting:** Denoise and smooth curves with a **low-pass filter** and **EMD**, then apply **NMS** to localize micro-expression segments.


---

## 7. Citation

If you use this code, please cite our manuscript:

**[Henian Yang, Shucheng Huang, Hualong Yu]**, “Facial Core Anchoring Triangle: Enhancing Micro-Expression Spotting through Geometric Alignment,” submitted to *The Visual Computer*.

(BibTeX will be provided after publication.)











