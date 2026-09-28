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

* The `AstroPTv3 repository <https://github.com/Smith42/AstroPTv3>`_ —
  see its README, ``astro/PLAN.md`` (phase plan) and ``astro/docs/adr/``
  (architecture decision records).
* ``astro/EXPERIMENTS.md`` in that repository records the measured
  loader/throughput evidence behind its data pipeline.
