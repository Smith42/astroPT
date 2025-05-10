Installation
===========

AstroPT requires Python 3.10 or later. 

Installation via pip
------------------

You can install AstroPT via pip from PyPI:

.. code-block:: bash

   pip install astropt

This is the recommended installation method for most users.

Installation from source using uv
------------------------------------

For developers or those who want the latest features, you can install from source via uv:

.. code-block:: bash

   # Install uv if you don't have it
   pip install uv
   
   # Clone the repository
   git clone https://github.com/smith42/astropt.git
   cd astropt
   
   # Install with uv
   uv sync
