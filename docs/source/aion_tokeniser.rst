Training with AION tokenisation
================================

This guide describes how to train AstroPT models using learned AION tokenisation instead of patch-based tokenisation.

Overview
--------

AstroPT supports two main tokenisation approaches for image data:

1. **Patch-based tokenisation** (``train.py``) - Splits images into fixed spatial patches
2. **AION tokenisation** (``train_aion.py``) - Uses a learned VQ-VAE tokenizer from the `AION project <https://github.com/PolymathicAI/aion>`_

AION tokenisation offers a more compressed representation by learning a discrete codebook of visual tokens, similar to how DALL-E or VQGAN work. Instead of treating each image patch as a continuous vector, AION maps image regions to discrete token IDs from a learned vocabulary.

Key differences from patch-based tokenisation
--------------------------------------------

**Tokenisation location:**
   - **Patch-based**: Happens in the dataset class, produces continuous vectors
   - **AION**: Happens in the dataloader's collate function, produces discrete token IDs

**Model architecture:**
   - **Patch-based**: Uses ``Encoder`` module to project continuous patches to embedding space
   - **AION**: Uses ``Embedder`` module (embedding table) to look up discrete tokens

**Modality configuration:**
   - **Patch-based**: ``vocab_size=0`` indicates continuous input
   - **AION**: ``vocab_size=10000`` indicates discrete tokens from a learned codebook

**Memory footprint:**
   - **Patch-based**: Full image data flows through the model
   - **AION**: Compressed token IDs flow through the model

Installation
------------

To use AION tokenisation, you'll need to install the AION library:

.. code-block:: bash

   uv sync --all-extras

Command-line usage
------------------

Single GPU
~~~~~~~

.. code-block:: bash

   uv run scripts/train_aion.py

Modality configuration for AION
--------------------------------

The key difference in configuration is the ``vocab_size`` parameter:

.. code-block:: python

   modalities = [
       ModalityConfig(
           name="images_aion",
           input_size=10000,  # Must == vocab_size
           patch_size=0,      # Placeholder, not used for discrete tokens
           loss_weight=1.0,
           embed_pos=True,
           pos_input_size=1,
           vocab_size=10000,  # Size of AION's learned codebook
       ),
   ]

When ``vocab_size > 0``, AstroPT knows to use an embedding table instead of a continuous encoder.

Dataset and dataloader setup
-----------------------------

AION tokenisation requires a different dataloader setup and dataset than patch-based tokenisation as the AION tokeniser has only been trained on raw astronomical imagery. Here we will use Legacy Survey imagery.

1. **Dataset**: Load raw image data from Hugging Face

   .. code-block:: python
   
      from datasets import load_dataset
      
      tds = load_dataset(
          "Smith42/legacysurvey_hsc_crossmatched",
          split="train",
          streaming=True,
      ).select_columns("legacysurvey_image").rename_column("legacysurvey_image", "image")

2. **Codec initialisation**: Load the pre-trained AION tokenizer

   .. code-block:: python
   
      from aion.codecs.image import ImageCodec
      from aion.modalities import LegacySurveyImage
      
      image_codec = ImageCodec.from_pretrained(
          "polymathic-ai/aion-base",
          modality=LegacySurveyImage
      )
      image_codec = image_codec.eval()

3. **Collate function**: Tokenise in batches during loading

   .. code-block:: python
   
      from functools import partial
      
      def collate_and_tokenise(batch, image_codec):
          """Batch tokenisation in main process"""
          # Stack all fluxes into a single batch tensor
          flux_batch = torch.stack([
              torch.tensor(item["image"]["flux"]) 
              for item in batch
          ])
          
          # Create single batched LegacySurveyImage
          batched_img = LegacySurveyImage(
              flux=flux_batch,
              bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
          )
          
          # Single encode call for entire batch
          tokens = image_codec.encode(batched_img)
          
          batch_size = len(batch)
          
          return {
              "images_aion": tokens.long(),
              "images_aion_positions": torch.stack([
                  torch.arange(tokens.shape[1]) for _ in range(batch_size)
              ])
          }
      
      collate_fn = partial(collate_and_tokenise, image_codec=image_codec)
      
      train_loader = DataLoader(
          tds,
          batch_size=batch_size,
          num_workers=num_workers,
          pin_memory=True,
          persistent_workers=True if num_workers > 0 else False,
          collate_fn=collate_fn,
      )

Loss function differences
-------------------------

AION tokens are discrete, so the loss function differs from patch-based training:

- **Patch-based**: Uses Huber loss (robust L1/L2 hybrid) for continuous predictions
- **AION**: Uses cross-entropy loss for discrete token classification

.. code-block:: python

   # In the model's forward pass
   if "aion" in mod_name:
       loss += F.cross_entropy(
           pred.reshape(-1, pred.size(-1)),
           target.reshape(-1),
       ) * mod_config.loss_weight
   else:
       loss += F.huber_loss(pred, target) * mod_config.loss_weight

Validation and visualisation
-----------------------------

During validation, AION tokens must be decoded back to images for visualisation:

.. code-block:: python

   @torch.no_grad()
   def validate(iter_num, out_dir):
       model.eval()
       
       # Get predictions
       B = gid.process_modes(next(vdl), modality_registry, device)
       with ctx:
           P, loss = model(B["X"], B["Y"])
           
           # Decode target tokens to images
           Yim = torch.cat((
               torch.zeros(B["Y"]["images_aion"].shape[0], 1), 
               B["Y"]["images_aion"].cpu(),
           ), dim=1)
           Yim = image_codec.decode(
               Yim,
               bands=["DES-G", "DES-R", "DES-Z"],
           )
           
           # Decode predicted tokens to images
           Pim = torch.cat((
               torch.zeros(B["Y"]["images_aion"].shape[0], 1), 
               torch.argmax(P["images_aion"], dim=-1).cpu(),
           ), dim=1)
           Pim = image_codec.decode(
               Pim,
               bands=["DES-G", "DES-R", "DES-Z"],
           )
           
           # Visualize
           clip_and_norm = lambda x: (torch.clamp(x, x.min(), x.quantile(0.99)) - x.min()) / (x.quantile(0.99) - x.min())
           
           for ax, p, y in zip(axs, Pim.flux, Yim.flux):
               ax[0].imshow(clip_and_norm(y.swapaxes(0, -1)))
               ax[1].imshow(clip_and_norm(p.swapaxes(0, -1)))

References
----------

- AION project: https://github.com/PolymathicAI/aion
- AION paper: https://arxiv.org/abs/2510.17960
- VQ-VAE paper: https://arxiv.org/abs/1711.00937
