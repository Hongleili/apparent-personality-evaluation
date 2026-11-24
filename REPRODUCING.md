# Reproduction and artifact layout

## Archived and portable code

The outputs/ hierarchy preserves the research workspace's relative paths.
The scripts are the executed research implementations, except for the one
documented local-path substitution. They are not a newly tested, general-purpose
training package. Several runners execute at import time, so inspect a runner
before importing it. Do not run an entire directory indiscriminately.

The results/ hierarchy contains read-only reference evidence. Scripts generally
write within outputs/; this separation prevents overwriting the public reference
tables during a reproduction attempt.

For a new experiment, work in a separate copy, record path/environment changes,
regenerate your own manifests and provenance, and retain the original frozen
protocols as reference. Do not replace hashes merely to make a different cache,
model, split or checkpoint pass an existing run's integrity checks.

## Data supplied separately

| Experiment | Required inputs not distributed here |
|---|---|
| ChaLearn feature extraction | Official training/validation/test videos, matching annotation files and any provider archive passwords |
| CLIP repeated heads and baselines | Original normalized per-frame feature cache and provenance; labeled official manifests |
| Source-separated CLIP/SigLIP 2 training | Matching 10,000-clip feature bundles, labeled manifests, parent inventories and original reference artifacts checked by each runner |
| CelebA | img_align_celeba.zip, list_attr_celeba.txt, list_eval_partition.txt; generated feature cache for repeated heads/prompt runs |
| Qwen full test | Official test videos, frozen labeled test.csv, matching reference feature metadata, checkpoint weights downloaded from the model provider |
| Full private-result verification | Saved per-clip predictions/responses and selected weights referenced by verification scripts |

ChaLearn annotation loading in chalearn_data.py disallows pickle globals and
checks the expected data schema. Never replace this with unrestricted
pickle.load for an untrusted file. The archive-staging code checks extraction
paths and coverage.

## Environments

Keep feature extraction and head-training environments separate. Recorded
versions are provenance, not a promise of bitwise equivalence on another platform:

- Source-separated head training: Python 3.12.14, NumPy 2.5.4, PyTorch 2.15.0,
  CPU, two Torch threads. See provenance/source_head_training_environment.json.
- SigLIP 2 extraction: package versions, processor and MPS float32 configuration
  are in provenance/siglip2/feature_software.json.
- Qwen: exact revision, packages, prompt, decoding, image limits and failure
  policy are in outputs/qwen_fulltest/protocol.json.
- CLIP/CelebA runners record resolved versions in each generated run configuration.
  Repeated-seed protocols retain original comparison settings. Dependencies
  include NumPy, PyTorch, Transformers, Pillow, OpenCV and, for CelebA metrics,
  scikit-learn.

No single global requirements file is supplied: mixing these historically
different environments would misrepresent the evaluated setup. The portable
evaluator has its own minimal requirements.

## Entry points and order

1. **ChaLearn CLIP:** inspect chalearn_data.stage_dataset, the run functions
   in experiment.py and the public functions in repeated_seeds.py.
   experiment.py contains the frozen encoder, prompts, frame selection,
   six heads and metrics. resume_cached.py verifies cache and code identity.
   The optional OneDrive downloader accepts a user-supplied authorized link;
   no sharing link is included.
2. **Linear baselines:** run_baselines.main(feature_run, manifests, output)
   fits only on training data and selects ridge alpha using validation MSE.
   baseline_uncertainty/analyse.py uses retained predictions.
3. **Frame ablation:** run_ablation.py uses the original feature cache and
   matching official manifests. See its function signature and protocol.
4. **Source-separated CLIP:** prepare_protocol.py creates the first frozen
   design, run_source_disjoint.py runs it, and prepare_extension.py /
   run_partition.py implement subsequent partitions. Preparation scripts
   intentionally refuse to overwrite a frozen protocol. Published membership
   is in split_assignments/SEED/CONDITION/*.csv; join to licensed labels by
   filename and preserve order.
5. **SigLIP 2:** inspect download_encoder.py, prepare_protocol.py,
   stage_media.py, extract_features.py, run_pipeline.py and run_partition.py,
   in that order. Set CHALEARN_SOURCE_DIR to your archive directory. This
   reuses exact CLIP video hashes and frame indices. Analysis follows all
   partitions; finalize_figure.py adjusts presentation only.
6. **CelebA:** run_celeba.main(data, output) performs the original reconstructed
   GPU experiment. run_repeat_seeds.py uses its fixed feature cache;
   run_prompts.py performs separately reported wording checks. The original
   approximately 100,000-image historical training subset was unavailable;
   this reconstruction uses all 162,770 official training images.
7. **Qwen:** pilot and validation scripts precede the frozen test protocol.
   run_fulltest.py --help describes final inference. Invalid generations
   remain failures; they are not repaired or silently retried. The analysis
   script audits retained raw outputs and compares unchanged CLIP predictions.

Some older Colab helpers contain /content/... runtime paths and non-secret run
hashes. These identify the evaluated layout, not portable defaults or cloud
access links. Set the Python import path to include outputs/colab_rerun and,
for Qwen development scripts, outputs/qwen_pilot when needed.

## Portable evaluator schema

The labeled manifest requires filename, source_prefix, openness,
conscientiousness, extraversion, agreeableness and neuroticism columns.
The raw neuroticism annotation is retained numerically and interpreted in the
emotional-stability direction, consistently with the study.

Predictions have the same filenames in the same order and five numeric score
columns. Files are named MODEL_test_predictions.csv. For source-cluster
contrasts supply ridge, training_median and cnn_lstm_seed42 through
cnn_lstm_seed46. Other model files are also scored. An existing output
directory is refused.

Run each partition separately using seeds 20261123 through 20261127.
The 99.5% marginal intervals apply to the ten-comparison family for one encoder.
Do not pair errors across different condition test sets or treat the 25 neural
fits as 25 independent datasets.

## Lightweight checks

From the repository root, with NumPy installed:

    python outputs/colab_rerun/test_data_metrics.py
    python outputs/colab_rerun/test_streaming.py
    python -m unittest discover -s outputs/colab_rerun -p 'test_*.py'

The streaming test requires /usr/bin/zip and uses artificial fixtures. These
checks exercise metrics, blocked pickle globals, nested archive staging and
cache-resume validation. They do not rerun GPU inference or establish new
scientific results.
