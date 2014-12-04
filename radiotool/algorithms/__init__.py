"""
.. module:: algorithms
    :synopsis: Various audio algorithms

.. moduleauthor:: Steve Rubin <srubin@cs.berkeley.edu>

"""

# from .composition import Composition
from .novelty import novelty
# fortran version... not using this anymore
# from .build_table import build_table
from .build_table_mem_efficient import build_table as build_table_mem_efficient
# from .par_build_table import build_table as par_build_table
from .build_table_full_backtrace import build_table as build_table_full_backtrace
import retarget
