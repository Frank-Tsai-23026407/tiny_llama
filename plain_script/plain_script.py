import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


# Helper function for RoPE (Rotary Positional Embeddings)
def __rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Helper function for RoPE (Rotary Positional Embeddings)
def __apply_rope(x, seq_len, head_dim, start_pos=0):
    # x shape: (B, H, S, D)
    input_dtype = x.dtype
    x = x.to(torch.float32)
    
    rope_theta = 10000.0

    # position = torch.arange(seq_len, dtype=torch.float, device=x.device)
    position = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float, device=x.device)

    # Create a frequency vector: (D/2)
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=x.device) / head_dim))

    # Compute angles: (S, D/2)
    angles = torch.outer(position, freqs)
    
    # Cos and Sin: (S, D/2) -> (S, D) by repeating
    # This expands the (D/2) dimensions to (D) dimensions 
    # cos_angles = torch.cos(angles).repeat_interleave(2, dim=-1) 
    # sin_angles = torch.sin(angles).repeat_interleave(2, dim=-1) 
    cos_angles = torch.cos(torch.cat([angles, angles], dim=-1)) 
    sin_angles = torch.sin(torch.cat([angles, angles], dim=-1)) 

    # Reshape to match x for broadcasting: (S, D) -> (1, 1, S, D)
    cos_angles = cos_angles.unsqueeze(0).unsqueeze(0)
    sin_angles = sin_angles.unsqueeze(0).unsqueeze(0)
        
    # Element-wise operation with broadcasting (B, H, S, D) * (1, 1, S, D)
    result = x * cos_angles + __rotate_half(x) * sin_angles
    return result.to(input_dtype) # Return with original dtype

# function: multi-query attention w/ rope
def mqa_rope(i, wq, wk, wv, wo, num_heads, num_kv_heads, get_attn_scores=False):
    # i: input tensor (batch_size, seq_len, model_dim)
    # wq, wk, wv: weight matrices for Q, K, V (e.g., [out_dim, in_dim])
    # wo: output weight matrix (model_dim, model_dim)
    
    batch_size, seq_len, model_dim = i.shape
    head_dim = model_dim // num_heads
    num_q_heads = num_heads
    num_kv_heads = num_kv_heads if num_kv_heads is not None else num_q_heads
    
    # FIX 1: Use the correct projection result dimensions for GQA
    # Q projection result: (B, S, model_dim)
    q_proj = torch.matmul(i, wq.t())
    # K/V projection results: (B, S, num_kv_heads * head_dim)
    k_proj = torch.matmul(i, wk.t())
    v_proj = torch.matmul(i, wv.t())

    # view: Splitting into Multiple Heads
    q = q_proj.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
    k = k_proj.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v_proj.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    q = __apply_rope(q, seq_len, head_dim)
    k = __apply_rope(k, seq_len, head_dim)
    
    # Repeat k and v heads if num_q_heads > num_kv_heads (Grouped Query Attention)
    k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    
    # Attention calculation
    attn_scores  = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply causal mask
    # The mask should be broadcastable to the batch dimension
    mask = torch.triu(torch.ones(seq_len, seq_len, device=i.device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))

    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
    attn_output  = torch.matmul(attn_weights, v)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
    o = torch.matmul(attn_output, wo.t()) # Final output
    if get_attn_scores:
        return o, attn_scores
    else:
        return o

# function: feed-forward network
def ffn_SwiGLU(i, w_gate, w_up, w_down):
    # i: input tensor (batch_size, seq_len, model_dim)
    # w_gate, w_up, w_down: weight matrices (e.g., [out_dim, in_dim])
    
    # Step 1: Linear transformation for gate and up projections
    gate = torch.matmul(i, w_gate.t())
    up = torch.matmul(i, w_up.t())
    
    # Step 2: Swish activation on the gate (SiLU is Swish-1)
    gate = torch.nn.functional.silu(gate)
    
    # Step 3: Element-wise multiplication of gate and up projections
    hidden_states = gate * up
    
    # Step 4: Down projection
    return torch.matmul(hidden_states, w_down.t())

# function: rmsnorm
def rmsnorm(x, weight, eps=1e-5):
    # x: input tensor (batch_size, seq_len, model_dim)
    # weight: (model_dim)
    input_dtype = x.dtype
    # The Llama implementation uses float32 for variance calculation
    x = x.to(torch.float32) 
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    # The final result is returned in the original dtype
    return weight * x.to(input_dtype)

# function: input embedding
def input_embedding(input_ids, wte):
    # input_ids: (batch_size, seq_len)
    # wte: (vocab_size, model_dim)
    return torch.nn.functional.embedding(input_ids, wte)

def lm_head(hidden_states, wte):
    # hidden_states: (batch_size, seq_len, model_dim)
    # wte: (vocab_size, model_dim)
    return torch.matmul(hidden_states, wte.t())

# function: forward pass of a tinyllama transformer block
def transformer_block(x, params, num_heads, num_kv_heads, get_attention_scores=False):
    
    # LayerNorm 1
    # The norm has a learnable weight parameter
    x_norm = rmsnorm(x, params['norm1_weight'])
    
    # Multi-Query Attention with RoPE
    attn_output, attn_scores = mqa_rope(
        x_norm,
        params['wq'], params['wk'], params['wv'], params['wo'],
        num_heads, num_kv_heads,
        get_attn_scores=True
    )
    
    # Residual connection 1
    x = x + attn_output
    
    # LayerNorm 2
    x_norm = rmsnorm(x, params['norm2_weight'])
    
    # Feed-Forward Network with SwiGLU activation
    ffn_output = ffn_SwiGLU(
        x_norm,
        params['w_gate'], params['w_up'], params['w_down']
    )
    
    # Residual connection 2
    x = x + ffn_output
    
    if get_attention_scores:
        return x, attn_scores
    else:
        return x
    
class transfomer_block_with_kv_cache:
    def __init__(self, params, num_heads, num_kv_heads, get_attention_scores=False):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.k_cache = None
        self.v_cache = None
        self.sequence_length = 0
        self.params = params
        self.get_attention_scores = get_attention_scores
        
        
    def forward(self, x, params):
        # LayerNorm 1
        x_norm = rmsnorm(x, params['norm1_weight'])
        
        # Q projection
        batch_size, seq_len, model_dim = x.shape
        head_dim = model_dim // self.num_heads
        q_proj = torch.matmul(x_norm, params['wq'].t())
        q = q_proj.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        q = __apply_rope(q[:, :, -1, :], seq_len, head_dim, self.sequence_length)
        
        # K/V projection
        k_proj = torch.matmul(x_norm, params['wk'].t())
        v_proj = torch.matmul(x_norm, params['wv'].t())
        k = k_proj.view(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        v = v_proj.view(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        k = __apply_rope(k[:, :, -1, :], seq_len, head_dim, self.sequence_length)
        
        # Update KV cache
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        
        # Repeat k and v heads if num_q_heads > num_kv_heads (Grouped Query Attention)
        k_expanded = self.k_cache.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v_expanded = self.v_cache.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Attention calculation
        attn_scores  = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask
        total_seq_len = self.k_cache.shape[2]
        mask = torch.triu(torch.ones(total_seq_len, total_seq_len, device=x.device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask[:, :, -seq_len:, :], float('-inf'))

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output  = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        o = torch.matmul(attn_output, params['wo'].t())
        x = x + o
        
        # LayerNorm 2
        x_norm = rmsnorm(x, params['norm2_weight'])
        
        # Feed-Forward Network with SwiGLU activation
        ffn_output = ffn_SwiGLU(
            x_norm,
            params['w_gate'], params['w_up'], params['w_down']
        )
        
        # Residual connection 2
        x = x + ffn_output
        
        if self.get_attention_scores:
            return x, attn_scores
        else:
            return x

# main 
def main():
    # Step 1: Load the tinyllama model & example input
    device = 'cpu'
    # Use torch.float32 for consistency and to avoid bfloat16 issues if not using a GPU
    dtype = torch.float32 
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype)

    example_input = '\n<|user|>:hello</s>\n<|assistant|>:'
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
    
    # step 6: Get ground truth logits from the original model
    with torch.no_grad():
        ground_truth_outputs = model(**model_inputs, output_hidden_states=True)
        ground_truth_logits = ground_truth_outputs.logits.to(dtype)
        ground_truth_latents = ground_truth_outputs.hidden_states # Input embeddings
        
    # compare hidden states
    print("Latents from your implementation (last token, first 10):")
    # print(x[0, -1, :10])
    print(x_list[23][0, -1, :10])
    print("\nLatents from Hugging Face model (last token, first 10):")
    print("size of ground_truth_latents: ", len(ground_truth_latents))
    # print(ground_truth_latents)
    print(ground_truth_latents[22][0, -1, :10])
    print(x_list[22][0, -1, :10] / ground_truth_latents[22][0, -1, :10])


    # Step 7: Compare the logits
    print("Logits from your implementation (last token, first 10):")
    print(my_logits[0, -1, :10])
    print("\nLogits from Hugging Face model (last token, first 10):")
    print(ground_truth_logits[0, -1, :10])

    # Check if they are close
    # Using a small tolerance (1e-3 or 1e-4 is standard for fp32)
    are_close = torch.allclose(my_logits, ground_truth_logits, atol=1e-4) 
    print(f"\nAre the logits close? {are_close}")
    
if __name__ == "__main__":
    main()