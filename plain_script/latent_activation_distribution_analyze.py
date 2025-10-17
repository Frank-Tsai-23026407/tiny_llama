# from ...plain_script.plain_script import *
from plain_script import *
import matplotlib.pyplot as plt
import numpy as np
import os

# example_input = '\n<|user|>:hello</s>\n<|assistant|>:'
example_input = '\n<|user|>:How is the weather today?</s>\n<|assistant|>:It is hot today.</s>\n<|user|>:What about tomorrow?</s>\n<|assistant|>:'
example_input = '\n<|user|>:How is the weather today?</s>\n<|assistant|>:It is'
example_input = '\n<|user|>:Who is the president of US now?</s>\n<|assistant|>:As of 2021, the president of the'


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
    for layer_idx in range(2, num_layers):
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
    x_list.append(x)
    
    # step 5: lm head [important: the lm_head do not use the same weights as the input embedding]
    output_logits = lm_head(x, model.get_output_embeddings().weight.to(dtype))
    predict_next_token(output_logits, tokenizer, top_k=5, print_topk=True)
    
    
    # Step 6: Analyze and plot the distribution of latent activations
    output_dir = "latent_activation_plots"
    os.makedirs(output_dir, exist_ok=True)
    layer_names_list = []



    # Get token strings and prepare data structure for ratio analysis
    token_ids = model_inputs['input_ids'][0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    sequence_length = len(tokens)
    max_rms_ratios_per_token = [[] for _ in range(sequence_length)]
    max_abs_mean_ratios_per_token = [[] for _ in range(sequence_length)]

    for i, x_latent in enumerate(x_list):
        if i == 0:
            layer_name = "Initial_Embedding"
        elif i <= num_layers:
            layer_name = f"Transformer_Block_{i-1}"
        else:
            layer_name = "Final_Norm"

        layer_names_list.append(f"{i:02d}_{layer_name}")
        
        # Calculate metrics for each token
        for token_idx in range(sequence_length):
            token_activations = x_latent[0, token_idx, :].detach().cpu().numpy()
            abs_max = np.max(np.abs(token_activations))
            rms = np.sqrt(np.mean(np.square(token_activations)))
            abs_mean = np.mean(np.abs(token_activations))
            max_rms_ratio = abs_max / rms if rms > 0 else 0
            max_rms_ratios_per_token[token_idx].append(max_rms_ratio)
            max_abs_mean_ratio = abs_max / abs_mean if abs_mean > 0 else 0
            max_abs_mean_ratios_per_token[token_idx].append(max_abs_mean_ratio)
            # print(f"Layer {i:02d} ({layer_name}), Token {token_idx} ('{tokens[token_idx]}'): Abs Max = {abs_max:.4f}, RMS = {rms:.4f}, Ratio = {ratio:.4f}")

        # The histogram plot will continue to show the last token's distribution for clarity
        last_token_activations = x_latent[0, -1, :].detach().cpu().numpy()
        abs_max = np.max(np.abs(last_token_activations))
        rms = np.sqrt(np.mean(np.square(last_token_activations)))
        max_rms_ratio = abs_max / rms if rms > 0 else 0
        abs_mean = np.mean(np.abs(last_token_activations))
        max_abs_mean_ratio = abs_max / abs_mean if abs_mean > 0 else 0

        plt.figure(figsize=(12, 6))
        plt.hist(last_token_activations, bins=100, alpha=0.6, color='skyblue', edgecolor='blue')
        plt.title(f'Last Token Activation Distribution - {layer_name} (Layer {i})')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.text(0.05, 0.95, f'Abs Max / RMS: {max_rms_ratio:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(os.path.join(output_dir, f'latent_dist_{i:02d}_{layer_name}.png'))
        plt.close()
        
    for i, x_latent in enumerate(x_list):
        if i == 0:
            layer_name = "Initial_Embedding"
        elif i <= num_layers:
            layer_name = f"Transformer_Block_{i-1}"
        else:
            layer_name = "Final_Norm"

        plt.figure(figsize=(12, 6))
        for token_idx in range(sequence_length):
            token_activations = x_latent[0, token_idx, :].detach().cpu().numpy()
            plt.plot(token_activations, alpha=0.7, label=f"Token {token_idx}: '{tokens[token_idx]}'")
        
        plt.title(f'Activation Values per Token - {layer_name} (Layer {i})')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')
        
        plt.grid(axis='y', alpha=0.75)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output_dir, f'latent_activation_values{i:02d}_{layer_name}.png'))
        plt.close()

    # Plot the summary of max/rms ratios for each token across all layers
    plt.figure(figsize=(15, 7))
    for token_idx in range(sequence_length):
        plt.plot(layer_names_list, max_rms_ratios_per_token[token_idx], marker='o', linestyle='-', alpha=0.7, label=f"Token {token_idx}: '{tokens[token_idx]}'")
    plt.title('Ratio of Abs Max to RMS of Activations per Token Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Abs Max / RMS Ratio')
    plt.xticks(rotation=90)
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_max_to_rms_ratio.png'))
    plt.close()
    
    # Plot the summary of max/abs_mean ratios for each token across all layers
    plt.figure(figsize=(15, 7))
    for token_idx in range(sequence_length):
        plt.plot(layer_names_list, max_abs_mean_ratios_per_token[token_idx], marker='o', linestyle='-', alpha=0.7, label=f"Token {token_idx}: '{tokens[token_idx]}'")
    plt.title('Ratio of Abs Max to Abs Mean of Activations per Token Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Abs Max / Abs Mean Ratio')
    plt.xticks(rotation=90)
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_max_to_abs_mean_ratio.png'))
    plt.close()
    
    # Plot attention score maps
    for layer_idx, attention_scores in enumerate(attention_score_list):
        # attention_scores shape: (batch, heads, seq_len, seq_len)
        # Average scores across heads for a single visualization per layer
        avg_attention_scores = attention_scores.mean(dim=1).squeeze(0).detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention_scores, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Attention Score')
        plt.title(f'Average Attention Score Map - Layer {layer_idx}')
        plt.xlabel('Key (Attended-to) Token Index')
        plt.ylabel('Query (Attending) Token Index')
        
        # Set token labels for axes
        tick_labels = [f"{i}: '{tok}'" for i, tok in enumerate(tokens)]
        plt.xticks(ticks=np.arange(sequence_length), labels=tick_labels, rotation=90)
        plt.yticks(ticks=np.arange(sequence_length), labels=tick_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attention_map_layer_{layer_idx:02d}.png'))
        plt.close()
    
    
    
    print(f"\nSaved {len(x_list)*2 + 1 + len(attention_score_list)} plots to '{output_dir}'.")
    
if __name__ == "__main__":
    main()