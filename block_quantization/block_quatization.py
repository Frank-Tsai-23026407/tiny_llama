import torch
import math

def block_floating_point_quantize(weight: torch.Tensor, block_size: int, mantissa_bits: int) -> torch.Tensor:
    """
    Quantizes a weight matrix using Block Floating Point (BFP) quantization.

    BFP quantizes the elements in a block by finding the maximum absolute value
    (which determines a shared exponent) and quantizing the mantissas.

    Args:
        weight (torch.Tensor): The input weight matrix (e.g., from a linear layer).
        block_size (int): The number of elements in each block.
        mantissa_bits (int): The number of bits used to quantize the mantissa.

    Returns:
        torch.Tensor: The reconstructed (de-quantized) weight matrix.
    """
    original_shape = weight.shape
    # Flatten the weight tensor for block processing
    flat_weight = weight.flatten()
    num_elements = flat_weight.numel()

    # 1. Padding to ensure the tensor is divisible by the block size
    padding_needed = (block_size - (num_elements % block_size)) % block_size
    if padding_needed > 0:
        flat_weight = torch.nn.functional.pad(flat_weight, (0, padding_needed))

    num_blocks = flat_weight.numel() // block_size
    
    # Reshape into blocks
    blocks = flat_weight.view(num_blocks, block_size)

    # 2. Determine the shared exponent (E) for each block
    
    # Find the maximum absolute value in each block (the characteristic of the block)
    max_abs_values = torch.amax(torch.abs(blocks), dim=1, keepdim=True)
    
    # Calculate the exponent E: E = ceil(log2(max_abs_value))
    # E will be the smallest integer such that 2^E >= max_abs_value
    # We add a small epsilon to handle log2(0) if the block is all zeros
    epsilon = 1e-9 
    
    # Use torch.ceil(torch.log2(x)) to find the exponent E
    # Note: torch.log2(0) is -inf, but since we used max_abs_values, only zero blocks are an issue, 
    # which the epsilon handles.
    E = torch.ceil(torch.log2(max_abs_values + epsilon))

    # 3. Calculate the shared scaling factor (S) for each block
    # S = 2^E. This is used to normalize the block elements into the mantissa range [-1, 1).
    S = torch.pow(2.0, E)

    # 4. Calculate the floating-point mantissa (M)
    M = blocks / S
    
    # 5. Quantize the mantissa (M)
    
    # Determine the number of quantization levels (Q_max) for the mantissa
    # Symmetric quantization: levels from -Q_max to Q_max
    Q_max = (1 << (mantissa_bits - 1)) - 1
    
    # Linearly map the floating-point mantissa M (range ~[-1, 1)) to integer levels
    # M_quant = round(M * Q_max)
    M_quant = torch.round(M * Q_max)
    
    # Clamp the quantized mantissa to the available integer range
    M_quant = torch.clamp(M_quant, -Q_max, Q_max)
    
    # 6. De-quantize (reconstruct) the weight
    
    # M_dequant = M_quant / Q_max
    M_dequant = M_quant / Q_max
    
    # Reconstruct the block: W_approx = M_dequant * S
    reconstructed_blocks = M_dequant * S
    
    # 7. Reshape and remove padding
    reconstructed_flat = reconstructed_blocks.flatten()
    
    # Remove the padding added earlier
    if padding_needed > 0:
        reconstructed_flat = reconstructed_flat[:-padding_needed]
        
    # Reshape back to the original matrix shape
    reconstructed_weight = reconstructed_flat.view(original_shape)

    return reconstructed_weight


if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Block Floating Point Quantization Example ---")

    # 1. Define parameters and a sample weight matrix
    BLOCK_SIZE = 16
    MANTISSA_BITS = 5 # Total bits per element = Mantissa Bits + Exponent Bits + 1 Sign Bit (assuming 4-5 bits for exponent)

    # Create a 2D weight tensor (e.g., from a Linear layer with 32 input features and 64 output features)
    # The elements are deliberately spread across different magnitude ranges.
    W_orig = torch.randn(4, 32) * 0.1 # Small values
    W_orig[:, 20:] *= 10.0 # Larger values in the second half of columns
    W_orig[0, :] = torch.arange(32) / 10.0 # Diverse row

    print(f"\nOriginal Weight Tensor Shape: {W_orig.shape}")
    print(f"BFP Parameters: Block Size={BLOCK_SIZE}, Mantissa Bits={MANTISSA_BITS}")
    print(f"Original Weight (Partial):\n{W_orig}\n...")

    # 2. Perform BFP Quantization
    W_quantized = block_floating_point_quantize(W_orig, BLOCK_SIZE, MANTISSA_BITS)

    print(f"\nQuantized/Reconstructed Weight (Partial):\n{W_quantized}\n...")

    # 3. Calculate the Mean Squared Error (MSE)
    mse = torch.mean((W_orig - W_quantized) ** 2)
    print(f"\nMean Squared Error (MSE) after BFP: {mse.item():.6f}")

    # 4. Show the effect on a specific block (first 16 elements of the first row)
    block_index = 0
    original_block = W_orig.flatten()[block_index*BLOCK_SIZE : (block_index+1)*BLOCK_SIZE]
    reconstructed_block = W_quantized.flatten()[block_index*BLOCK_SIZE : (block_index+1)*BLOCK_SIZE]
    
    max_val_original = torch.max(torch.abs(original_block))
    exponent = torch.ceil(torch.log2(max_val_original + 1e-9)).item()
    
    print(f"\nAnalysis of Block {block_index} (Size {BLOCK_SIZE}):")
    print(f"  Max Absolute Value in Block: {max_val_original:.4f}")
    print(f"  Shared Exponent (E): {exponent}")
    print(f"  Scaling Factor (S=2^E): {2**exponent:.4f}")
    print(f"  Original Block (First 4): {original_block[:4]}")
    print(f"  Reconstructed Block (First 4): {reconstructed_block[:4]}")
    print(f"  Error in Block (L1 Norm): {torch.sum(torch.abs(original_block - reconstructed_block)):.6f}")