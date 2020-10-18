# imports
import sys
import enum
import numpy as np
import pandas as pd
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# take in program params
# TODO: turn these into CLI args
execution_type = sys.argv[1]
input_filepath = sys.argv[2]
input_length = int(sys.argv[3])
output_filepath = sys.argv[4]
# execution_type = int(input("What kind of execution type do you want?\n"
#                         "Enter '1' for sequential.\n"
#                         "Enter '2' for parallel explicit.\n"
#                         "Enter '3' for parallel unified.\n"))
# input_filepath = input("Input file path: ")
# input_filelength = int(input("Input file length: "))
# output_filepath = input("Output file path: ")

# validate params
# TODO

# read in inputs, convert to numpy array
gate_inputs = pd.read_csv(input_filepath, header=None).values[:input_length]


class Gates(enum.Enum):
    AND = 0
    OR = 1
    NAND = 2
    NOR = 3
    XOR = 4
    XNOR = 5


# declare functions for performing simulations

def simulate_gate(gate_spec):
    x1, x2, gate = gate_spec
    if Gates(gate) == Gates.AND:
        return x1 and x2
    elif Gates(gate) == Gates.OR:
        return x1 or x2
    elif Gates(gate) == Gates.NAND:
        return not (x1 and x2)
    elif Gates(gate) == Gates.NOR:
        return not (x1 or x2)
    elif Gates(gate) == Gates.XOR:
        return (x1 or x2) and not (x1 and x2)
    elif Gates(gate) == Gates.XNOR:
        return (x1 and x2) or (not x1 and not x2)
    else:
        # ? throw exception?
        return None


# gates should be an array of
def simulate_sequential(gates):
    return np.array(list(map(simulate_gate, gates)))


def simulate_parallel_explicit(gates):
    # TODO: same as above, but parallel with explicit cuda memory allocation
    return None


def simulate_parallel_unified(gates):
    # TODO: same as above, but parallel with unified cuda memory allocation
    return None


def write_to_output(exec_type, gates, filepath):
    if exec_type == "sequential":
        simulate_sequential(gates).tofile(filepath, sep='\n')
    elif exec_type == "parallel_explicit":
        pd.DataFrame(simulate_parallel_explicit(gates)).to_csv(filepath)
    elif exec_type == "parallel_unified":
        pd.DataFrame(simulate_parallel_unified(gates)).to_csv(filepath)
    else:
        # ? throw exception?
        return


# perform simulations
write_to_output(execution_type, gate_inputs, output_filepath)
