# from ...plain_script.plain_script import *
from plain_script.plain_script import *
import argparse
from block_quantization.block_quatization import block_floating_point_quantize
from utils import *

class TinyLlamaMyModel:
    def __init__(self, model_name="TinyLlama/TinyLlama_v1.1", device='cpu', stop_criteria=None, dtype=torch.float32, apply_bfp=False, bfp_block_size=16, bfp_mantissa_bits=4):
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        
        self.attention_blocks = []
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            if apply_bfp:
                layer.input_layernorm.weight.data = block_floating_point_quantize(layer.input_layernorm.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.self_attn.q_proj.weight.data = block_floating_point_quantize(layer.self_attn.q_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.self_attn.k_proj.weight.data = block_floating_point_quantize(layer.self_attn.k_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.self_attn.v_proj.weight.data = block_floating_point_quantize(layer.self_attn.v_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.self_attn.o_proj.weight.data = block_floating_point_quantize(layer.self_attn.o_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.post_attention_layernorm.weight.data = block_floating_point_quantize(layer.post_attention_layernorm.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.mlp.gate_proj.weight.data = block_floating_point_quantize(layer.mlp.gate_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.mlp.up_proj.weight.data = block_floating_point_quantize(layer.mlp.up_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.mlp.down_proj.weight.data = block_floating_point_quantize(layer.mlp.down_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
            params = {
                'norm1_weight': layer.input_layernorm.weight.to(device).to(dtype),
                'wq': layer.self_attn.q_proj.weight.to(device).to(dtype),
                'wk': layer.self_attn.k_proj.weight.to(device).to(dtype),
                'wv': layer.self_attn.v_proj.weight.to(device).to(dtype),
                'wo': layer.self_attn.o_proj.weight.to(device).to(dtype),
                'norm2_weight': layer.post_attention_layernorm.weight.to(device).to(dtype),
                'w_gate': layer.mlp.gate_proj.weight.to(device).to(dtype),
                'w_up': layer.mlp.up_proj.weight.to(device).to(dtype),
                'w_down': layer.mlp.down_proj.weight.to(device).to(dtype)
            }
            self.attention_blocks.append(transfomer_block_with_kv_cache(params, self.num_heads, self.num_kv_heads, device=device))
        self.stop_criteria = stop_criteria

        if apply_bfp:
            self.model.model.norm.weight.data = block_floating_point_quantize(self.model.model.norm.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
            self.model.get_input_embeddings().weight.data = block_floating_point_quantize(self.model.get_input_embeddings().weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
            self.model.get_output_embeddings().weight.data = block_floating_point_quantize(self.model.get_output_embeddings().weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
            
    def single_step(self, inputs, return_latents=False):
        # assert torch.device(inputs['input_ids'].device) == self.device, f"Input IDs must be on the same device as the model. Input device: {torch.device(inputs['input_ids'].device)}, Model device: {self.device}"
        
        if return_latents:
            x_list = []
        
        # embedding
        x = input_embedding(inputs['input_ids'], self.model.get_input_embeddings().weight.to(self.dtype))
        if return_latents:
            x_list.append(x)
        
        # transformer blocks with KV cache
        for layer_idx in range(self.num_layers):
            x = self.attention_blocks[layer_idx].forward(x, self.attention_blocks[layer_idx].params)
            if return_latents:
                x_list.append(x)    
        
        # final rmsnorm
        x = rmsnorm(x, self.model.model.norm.weight.to(self.dtype))
        if return_latents:
            x_list.append(x)
        
        # lm head
        logits = lm_head(x, self.model.get_output_embeddings().weight.to(self.dtype))
        
        if return_latents:
            return logits, x_list
        else:
            return logits
    
    def reset_kv_cache(self):
        for block in self.attention_blocks:
            block.k_cache = None
            block.v_cache = None
            block.sequence_length = 0
            
    def generate(self, input_text, max_new_tokens=100):
        input_ids = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)
        generated_ids = input_ids.tolist()[0]
        current_input_ids = input_ids
        
        while True:
        
            # embedding
            x = input_embedding(current_input_ids, self.model.get_input_embeddings().weight.to(self.dtype))
            
            # transformer blocks with KV cache
            for layer_idx in range(self.num_layers):
                x = self.attention_blocks[layer_idx].forward(x, self.attention_blocks[layer_idx].params)
                
            # final rmsnorm
            x = rmsnorm(x, self.model.model.norm.weight.to(self.dtype))
            
            # lm head
            logits = lm_head(x, self.model.get_output_embeddings().weight.to(self.dtype))

            # Get the last token's logits
            next_token_logits = logits[:, -1, :]
            
            # Sample the next token (greedy approach for simplicity)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append to generated sequence
            generated_ids.extend(next_token_id.tolist()[0])
            
            # Check for stopping criteria
            if self.stop_criteria(torch.tensor([generated_ids]), None): # scores argument is not used by StopOnTokens
                break
                
            # Set current_input_ids for the next iteration to be just the newly generated token
            current_input_ids = next_token_id
            
            # Limit generation length to avoid infinite loops
            if len(generated_ids) > 100: # Max 100 new tokens
                break

        decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return torch.tensor([generated_ids])
    

def test():
    # Step 1: Load the tinyllama model & example input
    example_input = '\n<|user|>:hello</s>\n<|assistant|>:'
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
    model_inputs = tokenizer([example_input], return_tensors="pt").to('cpu')
    
    # run my model    
    my_model = TinyLlamaMyModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device='cpu',
        stop_criteria=StopOnTokens(),
        dtype=torch.float32
    )
    
    my_logits, x_list = my_model.single_step(model_inputs, return_latents=True)
    
    # ground truth generation
    with torch.no_grad():
        ground_truth_outputs = model(**model_inputs, output_hidden_states=True)
        ground_truth_logits = ground_truth_outputs.logits.to(torch.float32)
        ground_truth_latents = ground_truth_outputs.hidden_states
        
    # compare hidden states
    print("Latents from your implementation (last token, first 10):")
    # print(x[0, -1, :10])
    print(x_list[21][0, -1, :10])
    print("\nLatents from Hugging Face model (last token, first 10):")
    print("size of ground_truth_latents: ", len(ground_truth_latents))
    # print(ground_truth_latents)
    print(ground_truth_latents[21][0, -1, :10])
    print(x_list[21][0, -1, :10] / ground_truth_latents[21][0, -1, :10])


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
    
    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
    
    my_model = TinyLlamaMyModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=device,
        stop_criteria=StopOnTokens(),
        dtype=torch.float32
    )
    
    outputs = my_model.generate(
        input_formatting([], input_text),
        max_new_tokens=1024
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Input Text:\n", input_formatting([], input_text))
    print("Generated Text:\n", generated_text)