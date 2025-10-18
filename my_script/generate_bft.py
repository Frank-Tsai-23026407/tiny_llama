# standard imports
import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import numpy as np
import datasets

# Add the parent directory to the system path to allow for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# my imports
from block_quantization.block_quatization import block_floating_point_quantize
from plain_script.plain_script import *
from tinyllama_my import TinyLlamaMyModel
from utils import *

# set up configurations
device = 'cpu'
dtype = torch.float32  # Using bfloat16 for model weights
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# create bft model
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
my_model = TinyLlamaMyModel(
    model_name=model_name,
    device=device,
    stop_criteria=StopOnTokens(),
    dtype=torch.float32, 
    apply_bfp=True,
    bfp_block_size=16,
    bfp_mantissa_bits=4
)

# generate
def generate(history, input_text):
    
    # Example input
    input_text = input_formatting(history, input_text)

    # Generate output
    output_ids = my_model.generate(input_text, max_new_tokens=1024)
    
    # convert to text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tinyllama ground truth inference",
        description="Run inference on TinyLlama model in the full precision mode.",
    )
    parser.add_argument(
        "-i", "--input_text", type=str, help="Input text for the model.",
        default='Who is the president of US now?',
    )
    args = parser.parse_args()

    input_text = args.input_text
    history = []
    
    generated_output = generate(history, input_text)
    print("Generated Output:\n", generated_output)
    