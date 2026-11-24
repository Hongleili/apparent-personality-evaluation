# Evaluation-Dependent Model Rankings in Apparent Personality Prediction

Research code and aggregate results accompanying the manuscript by **Honglei Li,
Hashamm Khalid, Jialou Wang and Muhammad Talha**.
Correspondence: **Jialou Wang — jialou.wang@northumbria.ac.uk**.

This repository contains the completed study's research scripts, frozen protocols,
label-free split assignments, aggregate results and a portable prediction evaluator.
It is an audit and reproduction resource, not a one-command download-and-train
application. Dataset media, annotation values, cached embeddings, per-clip
predictions, generated responses and model weights are not redistributed.

## Download the code

[Download the complete research-code-results.zip archive](research-code-results.zip),
unzip it, and work inside the apparent-personality-evaluation folder. All paths
below refer to that extracted folder. It contains 48 Python scripts and 161 files
in total. The standalone [prediction evaluator](evaluate_partition.py) and
[its requirements](requirements-evaluation.txt) are also available directly here.
[ARCHIVE_SHA256.txt](ARCHIVE_SHA256.txt) verifies the archive; the included
SHA256SUMS.json verifies its individual files.

## What is included

| Experiment | Code directory under outputs/ | Evidence under results/ |
|---|---|---|
| ChaLearn frozen CLIP, six regression heads, five training seeds and paired uncertainty | colab_rerun, paired_uncertainty | five_seed_results, paired_uncertainty |
| Linear/ridge baselines and paired uncertainty | linear_baselines, baseline_uncertainty | Matching folders |
| Frame ablation | chalearn_frame_ablation | Matching folder |
| Source-separated CLIP and random-clip controls, five partitions | chalearn_source_disjoint, chalearn_source_replication | chalearn_source_replication |
| Matched SigLIP 2 second-encoder extension | siglip2_source_replication | Matching folder |
| CelebA six-attribute evaluation, repeated seeds and prompt sensitivity | celeba_rerun, celeba_prompt_sensitivity | Matching folders |
| Qwen pilot, validation and full official test | qwen_pilot, qwen_validation50, qwen_fulltest, qwen_fulltest_analysis | Validation trait metrics and full-test aggregates |

The five source-separated partitions and their size-matched random-clip controls
are shared by CLIP and SigLIP 2. The split_assignments/ folder contains filenames
and recording-prefix groups only, not trait labels or verified person identifiers.

## Completed results and interpretation

- The matched source-separation follow-up completed **100 neural fits per
  encoder**, spanning five partitions, two conditions, two heads and five
  training seeds.
- CNN–LSTM has lower mean absolute error than ridge on every random-clip control
  and higher error on every source-separated test for both encoders.
  SigLIP 2 FFNN is worse than ridge on **three of five** source-separated
  partitions; this weaker result must not be described as universal.
- Qwen attempted all **2,000** official test clips, with **1,945 valid outputs
  (97.25% coverage)**. Its **0.17254 MAE** is conditional on those valid outputs.
  CLIP comparisons use the same valid subset where specified.
- CelebA measures agreement with six attribute labels. These are recognition
  errors, not a measurement of generated hallucinations or a causal explanation
  of personality-prediction errors. The original GPU seed and the matched CPU
  repeated-seed evaluation are reported separately.

The source-separated extension was designed after inspecting earlier CLIP
results. It is exploratory, not prospectively preregistered. Partitions overlap
and are dependent; source prefixes are recording proxies, not verified
identities. Conditions use different test clips and do not estimate a causal
effect of source overlap. The two encoders' ten-contrast uncertainty families
are separate, not a single study-wide multiplicity correction.

## Reproduction

Read [REPRODUCING.md](REPRODUCING.md) before running the research scripts.
The simplest reusable entry point is the NumPy-only evaluator. Supply your own
licensed labels and saved predictions; use a new output directory:

    python -m pip install -r evaluation/requirements.txt
    python evaluation/evaluate_partition.py --manifest /path/to/labeled_test.csv --predictions /path/to/baselines /path/to/neural_predictions --output /path/to/new_audit --seed 20261123

The frozen protocols preserve their historical preparation status. A field
saying “prepared” describes when that protocol was written, not the present
completion state. Completed aggregate results and verification summaries are
under results/.

## Provenance and access

The included provenance/source_manifest.json records
original and published file hashes. The only edited experimental script is
stage_media.py: the author's local dataset path is replaced with the
CHALEARN_SOURCE_DIR environment variable. Its numerical processing is unchanged.
Other archived scripts retain their original runtime paths and integrity checks;
do not bypass checks to resume an incompatible run. SHA256SUMS.json covers the
public package.

Obtain datasets from their providers under the applicable terms:
[ChaLearn First Impressions](https://chalearnlap.cvc.uab.cat/dataset/24/description/)
and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Model checkpoints also retain their respective provider terms.

No author-code reuse licence has been selected for this release. Public
availability does not grant a licence to redistribute the datasets, and does
not imply that the manuscript has been accepted or published. Please contact
the corresponding author regarding reuse.

The authors report no funding and no competing interests.
