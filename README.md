# VEAT Quantifies Implicit Associations in Text-to-Video Generator Sora and Reveals Challenges in Bias Mitigation

VEAT (Video Embedding Association Test) and SC-VEAT (Single-Category VEAT) extend Embedding Association Tests to the video domain. In this repository, we provide the code for:

- extracting frame-level embeddings from generated videos using CLIP image encoder, mean-pooling them into a video embedding;
- computing association **effect sizes (Cohen’s d)** and **p-values** between target and attribute video sets.


## Data
Our dataset consists of **3,660 Sora-generated videos**, organized into **122 video sets with 30 videos per set**, covering both validation benchmarks and bias-evaluation conditions.  The collection includes (i) **non-social and social concept sets** adapted from classic WEAT/IAT-style stimuli (e.g., flowers vs. insects; European American vs. African American names; male vs. female terms) and **OASIS-derived category prompts** for validating valence directionality, and (ii) **race/gender face videos**, **17 occupations**, and **7 major awards** to quantify representational and valence-based bias and its alignment with real-world disparities.  Videos were generated using controlled prompt templates (e.g., “A video of …”; for human subjects, “A video of the face of a/an … on a gray background”) to reduce prompt sensitivity and isolate demographic signals; each video is **5 seconds** and generated with default platform settings, with prompts entered manually via the Sora interface.  For occupations and awards, we additionally generate parallel sets that append **explicit debiasing prompts** (two variants) to evaluate prompt-based mitigation effects, and we pair these sets with **2024 workforce/laureate demographics** compiled from BLS and award sources for correlation analyses. 

Data will be available upon request.

 ## Results
 In our results, we first **validate VEAT/SC-VEAT** by showing a strong correlation between SC-VEAT effect sizes and **human-rated valence** from OASIS (**r = 0.91**), and by replicating the expected **directionality and magnitude** of prior WEAT-style associations in both non-social and social concept scenarios.  We then find **substantial valence-based race and gender biases** in Sora’s outputs, and show that bias measured across **17 occupations and 7 awards** is not random: effect sizes **track real-world demographic disparities**, with strong positive correlations between gender effect sizes and occupational % male (**r = 0.93**) and between race effect sizes and occupational % white (**r = 0.83**).   Finally, we test **prompt-based mitigation** and observe a nuanced pattern: explicit debiasing prompts often move effect sizes for historically dominant groups closer to neutrality (e.g., reducing bias for male-/white-associated occupations and STEM awards), but can **exacerbate or reverse** associations in contexts already stereotypically tied to marginalized groups (e.g., some Black-associated occupations and the Nobel Peace Prize), highlighting the risks of naïvely applying debiasing prompts to text-to-video generation.  


## Citation

```bibtex
@inproceedings{sun2026veat,
  title={VEAT Quantifies Implicit Associations in Text-to-Video Generator Sora and Reveals Challenges in Bias Mitigation},
  author={Sun, Yongxu and Saxon, Michael and Yang, Ian and Gueorguieva, Anna-Maria and Caliskan, Aylin},
  booktitle={Second Conference of the International Association for Safe and Ethical Artificial Intelligence (IASEAI)},
  year={2026}
}
```
