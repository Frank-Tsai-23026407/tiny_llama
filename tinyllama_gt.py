import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import numpy as np
import datasets
from utils import input_formatting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

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
        
    inputs = tokenizer(input_formatting([], input_text), return_tensors="pt").to(device)
    
    with torch.no_grad():
        # auto-regressive generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            do_sample=False
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Input Text:\n", input_formatting([], input_text))
    print("Generated Text:\n", generated_text)
        