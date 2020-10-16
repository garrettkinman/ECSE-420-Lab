# imports
import enum
import numpy as np
import pandas as pd
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# take in program params
execution_type = int(input("What kind of exectution type do you want?\n"
                        "Enter '1' for sequential.\n"
                        "Enter '2' for parallel explicit.\n"
                        "Enter '3' for parallel unified.\n"))
input_filepath = input("Input file path: ")
input_filelength = int(input("Input file length: "))
output_filepath = input("Output file path: ")

# validate params
# TODO

# read in inputs, convert to numpy array
# TODO


class Gates(enum.Enum):
    AND = 0
    OR = 1
    NAND = 2
    NOR = 3
    XOR = 4
    XNOR = 5

# declare functions for performing simulations

def simulate_gate(x1, x2, gate):
    if gate == Gates.AND:
        return x1 and x2
    elif gate == Gates.OR:
        return x1 or x2
    elif gate == Gates.NAND:
        return not (x1 and x2)
    elif gate == Gates.NOR:
        return not (x1 or x2)
    elif gate == Gates.XOR:
        return (x1 or x2) and not (x1 and x2)
    elif gate == Gates.XNOR:
        return (x1 and x2) or ((not x1) and (not x2))
    else:
        # ? throw exception?
        return None

def simulate_sequential(gates):
    # TODO: for each gate, calculate the output, write to output array
    return None

def simulate_parallel_explicit(gates):
    # TODO: same as above, but parallel with explicit cuda memory allocation
    return None

def simulate_parallel_unified(gates):
    # TODO: same as above, but parallel with unified cuda memory allocation
    return None