REGISTRY = {}

from .basic_controller import BasicMAC
from .dicg_controller import DICGraphMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dicg_mac"] = DICGraphMAC