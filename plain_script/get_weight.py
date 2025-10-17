# from ...plain_script.plain_script import *
from plain_script import *
import matplotlib.pyplot as plt
import numpy as np
import os

# example_input = '\n<|user|>:hello</s>\n<|assistant|>:'
example_input = '\n<|user|>:How is the weather today?</s>\n<|assistant|>:It is hot today.</s>\n<|user|>:What about tomorrow?</s>\n<|assistant|>:'
example_input = '\n<|user|>:How is the weather today?</s>\n<|assistant|>:It is'
example_input = '\n<|user|>:Who is the president of US now?</s>\n<|assistant|>:As of 2021, the president of the'
example_input = '\n<|user|>:>:A man is seen bending down before a set of weights with others watching him on the side. the man\nA. then lifts the weights up around his head and drops them down while others stand on side of the chair and watch.\nB. then takes the weights and lifts himself up while others do the same on the sides.\nC. lifts up the weights over his head.\nD. then lifts up the weight over his head and throws it down towards others.\nAnswer:</s>\n<|assistant|>: Option'


def predict_next_token(logits, tokenizer, top_k=5, print_topk=False):
    # logits: (batch_size, seq_len, vocab_size)
    probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    next_token_id = torch.argmax(probs[:, -1, :], dim=-1)  # Get the token ID with the highest probability for the last token
    if print_topk:
        topk_probs, topk_indices = torch.topk(probs[:, -1, :], top_k)
        print(f"Top {top_k} token predictions: ", end="")
        for prob, idx in zip(topk_probs[0], topk_indices[0]):
            token = tokenizer.decode(idx.item())
            print(f"{token} {prob.item():.4f}", end=", ")
        print()
    return next_token_id.item()

def main():
    # Step 1: Load the tinyllama model & example input
    device = 'cpu'
    # Use torch.float32 for consistency and to avoid bfloat16 issues if not using a GPU
    dtype = torch.float32 
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype)

    
    model_inputs = tokenizer([example_input], return_tensors="pt").to(device)
    x_list = [] # To store intermediate latents for comparison later


    # Step 2: embedding
    x = input_embedding(model_inputs['input_ids'], model.get_input_embeddings().weight.to(dtype))
    x_list.append(x) # Store initial embeddings for comparison later
    attention_score_list = [] # To store attention scores if needed
    
    # Step 3: transformer blocks
    num_layers = model.config.num_hidden_layers
    for layer_idx in range(0, num_layers):
        layer = model.model.layers[layer_idx]
        params = {
            'norm1_weight': layer.input_layernorm.weight.to(dtype),
            'wq': layer.self_attn.q_proj.weight.to(dtype),
            'wk': layer.self_attn.k_proj.weight.to(dtype),
            'wv': layer.self_attn.v_proj.weight.to(dtype),
            'wo': layer.self_attn.o_proj.weight.to(dtype),
            'norm2_weight': layer.post_attention_layernorm.weight.to(dtype),
            'w_gate': layer.mlp.gate_proj.weight.to(dtype),
            'w_up': layer.mlp.up_proj.weight.to(dtype),
            'w_down': layer.mlp.down_proj.weight.to(dtype)
        }
        x, curr_attention_score = transformer_block(x, params, model.config.num_attention_heads, model.config.num_key_value_heads, get_attention_scores=True)
        x_list.append(x) # Store intermediate latents for comparison later
        attention_score_list.append(curr_attention_score) # Store attention scores if needed

        print("Layer {}/{}:".format(layer_idx+1, num_layers), end=" ")
        latent_logic = lm_head(x, model.get_output_embeddings().weight.to(dtype))
        predict_next_token(latent_logic, tokenizer, top_k=5, print_topk=True)
    
    # step 4: final rmsnorm
    x = rmsnorm(x, model.model.norm.weight.to(dtype))
    print(model.model.norm.weight.to(dtype))
    x_list.append(x)
    
    # step 5: lm head [important: the lm_head do not use the same weights as the input embedding]
    output_logits = lm_head(x, model.get_output_embeddings().weight.to(dtype))
    predict_next_token(output_logits, tokenizer, top_k=5, print_topk=True)
    
    # Step 7: Analyze and plot the correlation between input embedding and LM head weights
    input_embeddings = model.get_input_embeddings().weight.to(dtype).detach().cpu().numpy()
    lm_head_weights = model.get_output_embeddings().weight.to(dtype).detach().cpu().numpy()

    # 1. Overall correlation
    overall_corr = np.corrcoef(input_embeddings.flatten(), lm_head_weights.flatten())[0, 1]
    print(f"\nOverall correlation between input embedding and LM head weights: {overall_corr:.4f}")

    # 2. Per-token correlation
    per_token_corrs = [np.corrcoef(input_embeddings[i], lm_head_weights[i])[0, 1] for i in range(input_embeddings.shape[0])]
    
    plt.figure(figsize=(12, 6))
    plt.hist(per_token_corrs, bins=100, alpha=0.7, color='purple')
    plt.title('Distribution of Per-Token Correlation Coefficients')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join('correlation_per_token.png'))
    plt.close()

    # 3. Per-feature correlation
    per_feature_corrs = [np.corrcoef(input_embeddings[:, j], lm_head_weights[:, j])[0, 1] for j in range(input_embeddings.shape[1])]

    plt.figure(figsize=(12, 6))
    plt.hist(per_feature_corrs, bins=100, alpha=0.7, color='green')
    plt.title('Distribution of Per-Feature Correlation Coefficients')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join('correlation_per_feature.png'))
    plt.close()

    # 4. Scatter plot of a few token embeddings
    plt.figure(figsize=(10, 10))
    num_tokens_to_plot = 200
    indices = np.random.choice(input_embeddings.shape[0], num_tokens_to_plot, replace=False)
    plt.scatter(input_embeddings[indices, 0], lm_head_weights[indices, 0], alpha=0.5, label='Feature 0')
    plt.scatter(input_embeddings[indices, 1], lm_head_weights[indices, 1], alpha=0.5, label='Feature 1')
    plt.title(f'Scatter Plot of Input vs. LM Head Weights for {num_tokens_to_plot} Random Tokens')
    plt.xlabel('Input Embedding Weight Value')
    plt.ylabel('LM Head Weight Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('correlation_scatter.png'))
    plt.close()
    
if __name__ == "__main__":
    main()