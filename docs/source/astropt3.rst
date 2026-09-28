AstroPTv3
=========

*A sister line of AstroPT, built for the multimodal foundation-model regime.*

What is AstroPTv3?
------------------

AstroPTv3 is a from-scratch suite of multimodal astronomical foundation
models (70M–12B, Pythia-mirrored training schedules): a SmolLM3 decoder
body fed *continuous* image/spectrum patch tokens through per-modality
regression heads, pretrained on the Multimodal Universe (LegacySurvey,
DESI EDR SV3, HSC PDR3 Wide) streamed live from the Hugging Face hub via
LSDB — there is no local corpus copy.

The repository lives at https://github.com/Smith42/AstroPTv3.

How does it relate to AstroPT?
------------------------------

The two projects are co-evolving sister lines, not versions of one
programme:

* **AstroPT** (this repository) — nanoGPT lineage; the single-file
  training loop, the MLP tokeniser and the AION tokeniser experiments,
  BERT-style MAE, and the scaling/probing programme (COLM 2026) live
  here.
* **AstroPTv3** — Hugging Face ``smollm`` lineage; exact-likelihood
  patch regression with JetFormer/GIVT tokenisers, a transformers
  release artifact plus a nanotron pretraining stack, packed
  multimodal sequences (images, spectra, scalars) over multiple
  instruments, and live LSDB streaming of HATS catalogs.

AstroPTv3's JetFormer/GIVT tokeniser was ported from this repository's
``sogol_branch``. Findings cross over in both directions —
normalization conventions, evaluation practice, and checkpoint
schedules are shared lore — while the codebases stay separate.

Where to go next
----------------

* The `AstroPTv3 repository <https://github.com/Smith42/AstroPTv3>`_ —
  see its README, ``astro/PLAN.md`` (phase plan) and ``astro/docs/adr/``
  (architecture decision records).
* ``astro/EXPERIMENTS.md`` in that repository records the measured
  loader/throughput evidence behind its data pipeline.
