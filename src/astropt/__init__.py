import sys
from astropt.tokenizers import resnet_adapter
sys.modules['astropt.resnet_adapter'] = resnet_adapter

from astropt.tokenizers import aion_tokeniser
sys.modules['astropt.aion_tokeniser'] = aion_tokeniser
