from plain_script import *
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinyllama_my import TinyLlamaMyModel, StopOnTokens

import re

import datasets


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
    # Step 1: Load the tinyllama model & example input
    device = 'cpu'
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

    correct_predictions = 0
    total_questions = 0
    
    # use my model for evaluation
    my_model = TinyLlamaMyModel(
        model_name="TinyLlama/TinyLlama_v1.1",
        device='cpu',
        stop_criteria=StopOnTokens(),
        dtype=torch.float32
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
                choice_log_likelihoods = []
                for choice_text in choices:
                    full_text = context + " " + choice_text[0]
                    
                    if total_questions <= 5:
                        print(f"Full text for likelihood calculation:\n{full_text}\n")
                    
                    # Tokenize the full text
                    tokenized_input = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
                    input_ids = tokenized_input.input_ids
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
                    
                    # Sum the negative log-likelihoods for the choice part
                    # We need to identify which tokens correspond to the choice_text
                    # This is a simplified approach, assuming the choice starts after the context
                    context_tokens_len = tokenizer(context, return_tensors="pt", truncation=True).input_ids.shape[1]
                    # choice_loss = loss[context_tokens_len - 1:].sum() # -1 because labels are shifted
                    choice_loss = loss[context_tokens_len - 1:].mean()
                    
                    choice_log_likelihoods.append(-choice_loss.item()) # Store as log-likelihood

                # Predict the choice with the highest log-likelihood
                predicted_choice_idx = np.argmax(choice_log_likelihoods)
                predicted_answer_char = chr(65 + predicted_choice_idx)
                
                true_answer_char = chr(65 + true_label)

                if predicted_choice_idx == true_label:
                    correct_predictions += 1
                total_questions += 1
                
                if total_questions <= 5: # Print first 5 examples
                    print(f"\nContext:\n{context}")
                    for idx, ch in enumerate(choices):
                        print(f"{chr(65+idx)}. {ch} (Log-likelihood: {choice_log_likelihoods[idx]:.2f})")
                    print(f"True Answer: {true_answer_char}")
                    print(f"Predicted Answer: {predicted_answer_char}")
                    print("-" * 30)

    accuracy = correct_predictions / total_questions
    print(f"\n--- HellaSwag Evaluation Results ---")
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()