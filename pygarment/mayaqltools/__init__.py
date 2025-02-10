"""
    Package for to simulate garments from patterns in Maya with Qualoth. 

    Main dependencies:
        * Maya 2022+ (uses Python 3.6+)
        * Arnold Renderer
        * Qualoth (compatible with your Maya version)
    
    To run the package in Maya don't foget to add it to PYTHONPATH!
"""

from importlib import reload

import pygarment.mayaqltools.garmentUI as garmentUI
import pygarment.mayaqltools.mayascene as mayascene
import pygarment.mayaqltools.qualothwrapper as qualothwrapper
import pygarment.mayaqltools.scan_imitation as scan_imitation
import pygarment.mayaqltools.simulation as simulation
import pygarment.mayaqltools.utils as utils

from .mayascene import (
    MayaGarment,
    MayaGarmentWithUI,
    PatternLoadingError,
    Scene,
)

reload(mayascene)


reload(simulation)
reload(qualothwrapper)
reload(garmentUI)
reload(scan_imitation)
reload(utils)
