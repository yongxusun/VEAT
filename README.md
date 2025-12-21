# VEAT / SC-VEAT for Text-to-Video (Sora)

This repository contains the reference implementation used in **“VEAT Quantifies Implicit Associations in Text-to-Video Generator Sora and Reveals Challenges in Bias Mitigation”** and the corresponding paper PDF under `paper/`.

VEAT (Video Embedding Association Test) and SC-VEAT (Single-Category VEAT) extend Embedding Association Tests to the video domain by:

- extracting frame-level embeddings from generated videos using the **CLIP image encoder**, mean-pooling them into a video embedding;
- computing association **effect sizes (Cohen’s d)** and **permutation-test p-values** between target and attribute video sets.

> Paper: `paper/Sora_IASEAI_26.pdf`

## Repository structure

```
.
├── paper/
│   └── Sora_IASEAI_26.pdf
├── src/
│   ├── sora_implicit_association_test.py
│   └── compute_metrics.py
├── data/
├── results/
└── README.md
```


## Data prerequisites

At the time of writing, Sora did not provide a public API. Videos were generated through the Sora UI and downloaded manually (see Appendix A.2 in the paper).

The main script expects **folders of videos** (`.mp4`) for each target/attribute concept (e.g., `pleasant/`, `unpleasant/`, `man/`, `woman/`, etc.). In `src/sora_implicit_association_test_anonymized.py`, folder paths are currently placeholders (`"anonymized_directory"`). Update those paths to point to your local folders.

Recommended local layout:

```
data/
  videos/
    pleasant/
    unpleasant/
    flower/
    insect/
    ...
```

## Running VEAT / SC-VEAT

What it does (high level):
1. Extracts video embeddings by sampling frames every 0.25 seconds and encoding each frame with CLIP, then mean-pooling across frames.
2. Runs **WEAT-style** two-target/two-attribute tests (VEAT) and **single-category** tests (SC-VEAT).
3. Prints effect sizes and permutation-test p-values.

## Citation

If you use this code or build on VEAT/SC-VEAT, please cite the paper:

```bibtex
@inproceedings{sun2026veat,
  title={VEAT Quantifies Implicit Associations in Text-to-Video Generator Sora and Reveals Challenges in Bias Mitigation},
  author={Sun, Yongxu and Saxon, Michael and Yang, Ian and Gueorguieva, Anna-Maria and Caliskan, Aylin},
  booktitle={Second Conference of the International Association for Safe and Ethical Artificial Intelligence (IASEAI)},
  year={2026}
}
```
