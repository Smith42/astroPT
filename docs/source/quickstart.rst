Quickstart
=========

This guide will help you get started with AstroPT quickly.

Loading a pre-trained model
--------------------------

To load and run a pre-trained AstroPT model from Hugging Face 🤗, use the ``load_astropt`` function:

.. code-block:: python

   from astropt.model_utils import load_astropt

   model, model_args = load_astropt(
       repo_id="smith42/astropt_sparse",
       path="astropt/p16k10",
       weights_filename="ckpt.pt",
   )
   model = model.to("cuda")  # Move to GPU if available

Available pre-trained models
--------------------------

Below are some pre-trained models you can load with the code snippet above:

**DESI Legacy Survey Model**

- **Modalities:** JPG galaxy imagery
- **AstroPT version:** v1.0.0
- **Model weights:** `AstroPT <https://huggingface.co/Smith42/astroPT>`_
- **Dataset:** `Galaxies Dataset <https://huggingface.co/datasets/Smith42/galaxies>`_
- **Paper:** `arXiv:2405.14930 <https://arxiv.org/abs/2405.14930>`_

**Euclid Model**

- **Modalities:** FITS VIS, NISP galaxy imagery and SED data
- **AstroPT version:** v1.0.2
- **Model weights:** `AstroPT-Euclid <https://huggingface.co/collections/msiudek/astropt-euclid-67d061928ac0a447265ac8b8>`_
- **Dataset:** `Euclid Training Dataset <https://huggingface.co/datasets/msiudek/astroPT_euclid_training_dataset>`_
- **Paper:** `arXiv:2503.15312 <https://arxiv.org/abs/2503.15312>`_

Basic model usage
---------------

Here's a simple example of how to use the model for inference:

.. code-block:: python

   import torch
   from astropt.model_utils import load_astropt
   
   # Load the model
   model, model_args = load_astropt(
       repo_id="smith42/astropt_sparse",
       path="astropt/p16k10",
       weights_filename="ckpt.pt",
   )
   model = model.to("cuda")
   
   # Prepare your input data
   # This is a simplified example - actual preprocessing depends on your data
   input_data = preprocess_your_data(your_data)
   
   # Convert to tensor and move to the same device as the model
   input_tensor = torch.tensor(input_data).to("cuda")
   
   # Run inference
   with torch.no_grad():
       output = model(input_tensor)
   
   # Process the output
   processed_output = process_output(output)
