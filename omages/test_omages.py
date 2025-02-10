""" 
This is an interactive notebook for previewing the omages dataset.
For more information for using .py file as jupyter notebooks, please refer to:
https://code.visualstudio.com/docs/python/jupyter-support-py

"""

from ipynb_header import *
from xgutils import omgutil

example_omage_1024 = "assets/B0742FHDJF_objamage_tensor_1024.npz"
omage = omgutil.load_omg(example_omage_1024)
vomg, rdimg, _ = omgutil.preview_omg(omage)

omgutil.omg2object(omage)
