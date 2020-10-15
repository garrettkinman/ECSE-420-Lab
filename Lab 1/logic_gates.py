# imports
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
