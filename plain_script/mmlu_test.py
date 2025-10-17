from plain_script import *
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np



def main():
    # Step 1: Load the tinyllama model & example input
    device = 'cpu'
    # Use torch.float32 for consistency and to avoid bfloat16 issues if not using a GPU
    dtype = torch.float32 
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype)
    model.eval() # Set model to evaluation mode

    # Load MMLU dataset
    # For demonstration, let's use a small subset or a specific subject
    # You might need to install 'evaluate' and 'datasets' libraries
    # pip install evaluate datasets
    dataset = load_dataset("cais/mmlu", "all", split="test")
    
    # Filter for a specific subject or a smaller sample for quick testing
    # For example, let's take a small sample from the first subject
    # subjects = dataset.features['subject'].names
    # print(f"Available MMLU subjects: {subjects}")
    
    # Let's pick one subject for testing, e.g., "abstract_algebra"
    # filtered_dataset = dataset.filter(lambda x: x["subject"] == "abstract_algebra")
    # Or just take a small sample from the full dataset
    sample_size = 100 # Adjust as needed for testing
    dataset_sample = dataset.shuffle(seed=42).select(range(sample_size))

    # Function to format MMLU questions for TinyLlama
    def format_question(example):
        question = example["question"]
        choices = example["choices"]
        
        # Format as a chat-like prompt
        prompt = f"\n<|user|>:{question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:</s>\n<|assistant|>:"
        return {"prompt": prompt, "answer": example["answer"]}

    formatted_dataset = dataset_sample.map(format_question)

    # DataLoader for batching
    batch_size = 1 # Process one question at a time for simplicity
    data_loader = DataLoader(formatted_dataset, batch_size=batch_size)

    correct_predictions = 0
    total_questions = 0

    for batch in tqdm(data_loader, desc="Evaluating MMLU"):
        prompts = batch["prompt"]
        true_answers = batch["answer"]

        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        # log model inputs for debugging to file
        with open("mmlu_debug_inputs.txt", "a") as f:
            f.write(f"Input IDs: {model_inputs['input_ids']}\n")
            f.write(f"Attention Mask: {model_inputs['attention_mask']}\n")
            f.write(f"Prompts: {prompts}\n")
            f.write(f"True Answers: {true_answers}\n\n")
        
        # Get ground truth logits from the original model
        with torch.no_grad():
            # Use the original model's forward pass to get logits
            outputs = model(**model_inputs)
            ground_truth_logits = outputs.logits

            # Predict the next token
            # We are interested in the logit for 'A', 'B', 'C', 'D'
            # The tokenizer might tokenize 'A' as multiple tokens, so we need to be careful.
            # For simplicity, let's assume single token answers 'A', 'B', 'C', 'D'
            
            # Get the logits for the last token in the sequence
            last_token_logits = ground_truth_logits[:, -1, :]

            # Get token IDs for 'A', 'B', 'C', 'D'
            token_a_id = tokenizer.convert_tokens_to_ids('A')
            token_b_id = tokenizer.convert_tokens_to_ids('B')
            token_c_id = tokenizer.convert_tokens_to_ids('C')
            token_d_id = tokenizer.convert_tokens_to_ids('D')

            choice_ids = [token_a_id, token_b_id, token_c_id, token_d_id]
            
            # Extract logits for these choices
            choice_logits = last_token_logits[:, choice_ids]

            # Predict the choice with the highest logit
            predicted_choice_idx = torch.argmax(choice_logits, dim=-1).item()
            predicted_answer_char = chr(65 + predicted_choice_idx)
            
            # Convert true answer index (0, 1, 2, 3) to char ('A', 'B', 'C', 'D')
            true_answer_char = chr(65 + true_answers.item())

            if predicted_answer_char == true_answer_char:
                correct_predictions += 1
            total_questions += 1
            
            # Optional: Print some examples
            # if total_questions <= 5:
            #     print(f"\nPrompt:\n{prompts[0]}")
            #     print(f"True Answer: {true_answer_char}")
            #     print(f"Predicted Answer: {predicted_answer_char}")
            #     print("-" * 30)

    accuracy = correct_predictions / total_questions
    print(f"\n--- MMLU Evaluation Results ---")
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()