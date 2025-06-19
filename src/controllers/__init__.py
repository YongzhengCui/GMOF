REGISTRY = {}

from .basic_controller import BasicMAC
from .gmof_controller import GMOF_MAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["gmof_mac"] = GMOF_MAC
