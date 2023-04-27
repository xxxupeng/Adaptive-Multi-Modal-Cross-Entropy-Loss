from models.PSMNet.stackhourglass import PSMNet
from models.GwcNet.gwcnet import GwcNet_G, GwcNet_GC
from models.GANet.GANet_deep import GANet



__models__ = {
    "PSMNet": PSMNet,
    "GwcNet_G": GwcNet_G,
    "GwcNet_GC": GwcNet_GC,
    "GANet":GANet, 
    }
