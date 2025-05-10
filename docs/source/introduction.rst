Introduction
===========

What is AstroPT?
--------------

AstroPT is a Large Observation Model (LOM) for astronomy - a transformer-based foundation model designed to understand and generate astronomical data. 

How does AstroPT work?
--------------------

AstroPT is an autoregressive transformer under the hood.

Similarly to language models that predict the next word in a sentence, AstroPT processes sequences of astronomical data chunks to predict what comes next.

The intuition here is that this next-token-prediction task requires the model to internalise some understanding of the physical processes underlying the training data. 

This is just like how a text GPT needs to have some knowledge of geography to guess a country's capital given a description of that country, or some knowledge of coding to write compilable Fortran.

Below we can see this principle applied to a galaxy image, where we split the image into chunks and pass them into an AstroPT model:

.. image:: /images/galaxy_im.png
   :width: 25%
   :alt: Galaxy image
.. image:: /images/apt.png
   :width: 74%
   :alt: AstroPT architecture

Of course we can apply this next-token-prediction task across many modalities due to its flexibility.

Check out `our work on Euclid data <https://arxiv.org/abs/2503.15312>`_ for an example, where we chain galaxy image tokens and spectral energy distribution data and pass them into a single, unified AstroPT model.

Key features
-----------

- **Multi-modal data support**: Works with galaxy images, spectral energy distributions, and more
- **Flexible architecture**: Based on the transformer architecture for powerful sequence modeling
- **Pre-trained models**: Available for DESI Legacy Survey and Euclid data
- **Easy integration**: Simple API for loading and using pre-trained models
- **Open Source**: MIT licensed for both academic and commercial use
