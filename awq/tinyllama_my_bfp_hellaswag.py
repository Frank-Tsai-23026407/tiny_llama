# standard libraries
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import re
import datasets

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# my libraries
from plain_script import *
from utils import *
from tinyllama_my import TinyLlamaMyModel


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)

def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyLlama on HellaSwag with optional BFP quantization.")
    parser.add_argument("--bft", action="store_true", help="Apply Block Floating Point quantization.")
    parser.add_argument("--m_bit", type=int, default=4, help="Mantissa bits for BFP quantization.")
    parser.add_argument("--b_size", type=int, default=16, help="Block size for BFP quantization.")
    
    args = parser.parse_args()

    apply_bfp = args.bft
    bfp_mantissa_bits = args.m_bit
    bfp_block_size = args.b_size

    if apply_bfp:
        print(f"Applying BFP quantization with block_size={bfp_block_size} and mantissa_bits={bfp_mantissa_bits}")
    else:
        print("Running without BFP quantization.")

    # Step 1: Load the tinyllama model & example input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Use torch.float32 for consistency and to avoid bfloat16 issues if not using a GPU
    dtype = torch.float32 
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=dtype)
    model.eval() # Set model to evaluation mode

    # Load HellaSwag dataset
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    dataset_sample = dataset
    dataset_sample = process_docs(dataset_sample)

    # Function to format HellaSwag questions for TinyLlama
    def format_question(example):
        # HellaSwag context and endings
        # For likelihood evaluation, we need to provide the full text including the correct ending
        # and also the context for each choice to calculate its likelihood.
        context = example["query"]
        endings = example["choices"]
        
        # Store each choice as a separate entry for likelihood calculation
        # The 'text' field will be the full sequence for likelihood calculation
        # The 'label' field is the index of the correct answer
        return {
            "context": context,
            "choices": endings,
            "label": example["gold"]}
    formatted_dataset = dataset_sample.map(format_question)


    # DataLoader for batching
    batch_size = 1 # Process one question at a time for simplicity
    data_loader = DataLoader(formatted_dataset, batch_size=batch_size)

    correct_predictions_sum = 0
    correct_predictions_mean = 0
    total_questions = 0
    
    # use my model for evaluation
    my_model = TinyLlamaMyModel(
        model_name="TinyLlama/TinyLlama_v1.1",
        device=device,
        stop_criteria=StopOnTokens(),
        dtype=dtype,
        apply_bfp=apply_bfp,
        bfp_block_size=bfp_block_size,
        bfp_mantissa_bits=bfp_mantissa_bits
    )


    for batch in tqdm(data_loader, desc="Evaluating HellaSwag"):
        contexts = batch["context"]
        choices_list = batch["choices"] # This will be a list of lists of strings
        true_labels = batch["label"]

        # Get ground truth logits from the original model
        with torch.no_grad():
            for i in range(len(contexts)): # Iterate through batch (batch_size is 1 here)
                context = contexts[i]
                choices = [c for c in choices_list] # Flatten choices
                true_label = true_labels[i].item()

                # Calculate log-likelihood for each choice
                choice_log_likelihoods_sum = []
                choice_log_likelihoods_mean = []
                for choice_text in choices:
                    full_text = context + " " + choice_text[0]
                    
                    if total_questions <= 5:
                        print(f"Full text for likelihood calculation:\n\"{full_text}\"\n")
                    
                    # Tokenize the full text
                    tokenized_input = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
                    input_ids = tokenized_input.input_ids
                    # print("input ids: ", input_ids)
                    attention_mask = tokenized_input.attention_mask

                    # Get model outputs
                    # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    outputs = my_model.single_step(tokenized_input)
                    my_model.reset_kv_cache()
                    
                    # Calculate the negative log-likelihood (NLL)
                    # Shift logits and labels for language modeling
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # Calculate loss for each token
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    # --- Determine the correct NLL Summation Window ---
                    
                    # 1. Get the number of tokens in the context (includes <bos>)
                    context_tokens_len = tokenizer(context, return_tensors="pt", truncation=True).input_ids.shape[1]
                    
                    # 2. Slice the loss tensor starting at context_tokens_len.
                    # This is the empirical correction: it SKIPS the NLL of the first token
                    # of the continuation (which is often a space/boundary token) to match lm-eval.
                    # The total loss array length is N-1. Index context_tokens_len-1 is the first
                    # continuation token's loss. Index context_tokens_len is the second.
                    choice_loss_continuation = loss[context_tokens_len:].sum()
                    
                    # --- 3. Calculate Scores ---
                    
                    # Score for 'acc' (Unnormalized Log-Likelihood)
                    # choice_loss_sum is the NLL sum for the continuation
                    choice_loss_sum = choice_loss_continuation
                    
                    # 4. Calculate the byte length of the choice text (e.g., using UTF-8)
                    choice_text_str = choice_text[0]
                    choice_byte_length = len(choice_text_str.encode('utf-8'))

                    # Score for 'acc_norm' (Byte-Normalized Log-Likelihood)
                    if choice_byte_length > 0:
                        # choice_loss_mean is the Normalized NLL: (NLL Sum / Byte Length)
                        choice_loss_mean = choice_loss_continuation.item() / choice_byte_length
                    else:
                        # Handle the case of an empty choice (assign a very high NLL)
                        choice_loss_mean = float('inf')
                    
                    # Append Log-Likelihood Scores (Log-Likelihood = -NLL)
                    choice_log_likelihoods_sum.append(-choice_loss_sum.item())
                    choice_log_likelihoods_mean.append(-choice_loss_mean)
                    
                # Predict the choice with the highest log-likelihood
                predicted_choice_idx_sum = np.argmax(choice_log_likelihoods_sum)
                predicted_choice_idx_mean = np.argmax(choice_log_likelihoods_mean)
                predicted_answer_char_sum = chr(65 + predicted_choice_idx_sum)
                predicted_answer_char_mean = chr(65 + predicted_choice_idx_mean)
                
                true_answer_char = chr(65 + true_label)

                if predicted_choice_idx_sum == true_label:
                    correct_predictions_sum += 1
                if predicted_choice_idx_mean == true_label:
                    correct_predictions_mean += 1
                total_questions += 1
                
                if total_questions <= 5: # Print first 5 examples
                    print(f"\nContext:\n{context}")
                    for idx, ch in enumerate(choices):
                        print(f"{chr(65+idx)}. {ch} (Log-likelihood: {choice_log_likelihoods_sum[idx]:.2f} (Sum) {choice_log_likelihoods_mean[idx]:.2f} (Mean))")
                    print(f"True Answer: {true_answer_char}")
                    print(f"Predicted Answer: {predicted_answer_char_sum} (Sum), {predicted_answer_char_mean} (Mean)")
                    print("-" * 30)

    accuracy_sum = correct_predictions_sum / total_questions
    accuracy_mean = correct_predictions_mean / total_questions
    print(f"\n--- HellaSwag Evaluation Results ---")
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions: {correct_predictions_sum} (Sum), {correct_predictions_mean} (Mean)")
    print(f"Accuracy: {accuracy_sum:.4f} (Sum), {accuracy_mean:.4f} (Mean)")


if __name__ == "__main__":
    main()