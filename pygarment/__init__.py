"""
    A Python library for building parametric sewing pattern programs
"""

# Operations
import pygarment.garmentcode.operators as ops
import pygarment.garmentcode.utils as utils

# Building blocks
from pygarment.garmentcode.component import Component
from pygarment.garmentcode.connector import Stitches
from pygarment.garmentcode.edge import (
    CircleEdge,
    CurveEdge,
    Edge,
    EdgeSequence,
)
from pygarment.garmentcode.edge_factory import (
    CircleEdgeFactory,
    CurveEdgeFactory,
    EdgeFactory,
    EdgeSeqFactory,
)
from pygarment.garmentcode.interface import Interface
from pygarment.garmentcode.panel import Panel

# Parameter support
from pygarment.garmentcode.params import BodyParametrizationBase, DesignSampler

# Errors
from pygarment.pattern.core import EmptyPatternError
