# from ...plain_script.plain_script import *
from plain_script import *

def main():
    # Step 1: Load the tinyllama model & example input
    device = 'cpu'
    # Use torch.float32 for consistency and to avoid bfloat16 issues if not using a GPU
    dtype = torch.float32 
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype)

    example_input = '\n<|user|>:How is the weather today?</s>\n<|assistant|>:It is hot today.</s>\n<|user|>:What about tomorrow?</s>\n<|assistant|>:'
    model_inputs = tokenizer([example_input], return_tensors="pt").to(device)
    x_list = [] # To store intermediate latents for comparison later


    # Step 2: embedding
    x = input_embedding(model_inputs['input_ids'], model.get_input_embeddings().weight.to(dtype))
    x_init = x.clone() # Save initial embeddings for comparison later
    x_list.append(x) # Store initial embeddings for comparison later
    
    # Step 3: transformer blocks
    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
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
        x = transformer_block(x, params, model.config.num_attention_heads, model.config.num_key_value_heads)
        x_list.append(x) # Store intermediate latents for comparison later
    
    # step 4: final rmsnorm
    x = rmsnorm(x, model.model.norm.weight.to(dtype))
    x_list.append(x)
    
    # step 5: lm head [important: the lm_head do not use the same weights as the input embedding]
    my_logits = lm_head(x, model.get_output_embeddings().weight.to(dtype))
    
if __name__ == "__main__":
    main()