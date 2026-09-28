AstroPTv3
=========

*AstroPT but make it so you don't need to download 100TB of stuff*

What is AstroPTv3?
------------------

AstroPTv3 is a from-scratch suite of multimodal astronomical foundation
models: a SmolLM3 decoder body fed *continuous* image/spectrum patch tokens 
through per-modality regression heads, pretrained on the Multimodal Universe
streamed live from the Hugging Face hub via LSDB, with no local corpus copy.

The repository lives at https://github.com/Smith42/AstroPTv3.

How does it relate to AstroPT?
------------------------------

The two projects are co-evolving:

* **AstroPT** (this repository) — nanoGPT lineage; the single-file
  training loop, the MLP tokeniser and the AION tokeniser experiments,
  BERT-style MAE, and the scaling/probing programme live here.
* **AstroPTv3** — Hugging Face ``smollm`` lineage; exact-likelihood
  patch regression with JetFormer/GIVT tokenisers, a transformers
  release artifact plus a nanotron pretraining stack, packed
  multimodal sequences (images, spectra, scalars) over multiple
  instruments, and live LSDB streaming of HATS catalogs.

AstroPTv3's JetFormer/GIVT tokeniser was ported from Sogol's great
work on ``sogol_branch``.

Where to go next
----------------

The complete and evolving AstroPTv3 documentation lives in its
repository:

* `Project README <https://github.com/Smith42/AstroPTv3/blob/main/astro/README.md>`_
* `Phase plan <https://github.com/Smith42/AstroPTv3/blob/main/astro/PLAN.md>`_
* `Training guide <https://github.com/Smith42/AstroPTv3/blob/main/astro/docs/training.md>`_
* `Architecture decisions <https://github.com/Smith42/AstroPTv3/tree/main/astro/docs/adr>`_
* `Experiments and benchmarks <https://github.com/Smith42/AstroPTv3/blob/main/astro/EXPERIMENTS.md>`_

This Read the Docs page is a short introduction; the AstroPTv3 repository
is the source of truth for its implementation and full documentation.
