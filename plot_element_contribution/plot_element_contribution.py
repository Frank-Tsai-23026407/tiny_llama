import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

# Step 1: Load the tinyllama model & example input
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

example_input = '\n<|user|>:hello</s>\n<|assistant|>:'
model_inputs = tokenizer([example_input], return_tensors="pt").to(device).to(model.dtype)


# Step 2: Plot all the weights of the model
import matplotlib.pyplot as plt
import numpy as np

def plot_model_weights(model):
    weights_matrix = []
    weights = []
    layer_names = []

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:  # Only consider weight matrices
            weights_matrix.append(param.detach().cpu().numpy())
            weights.append(param.detach().cpu().numpy().flatten())
            layer_names.append(name)

    for i, weight in enumerate(weights):
        #
        
        # subplot 1: weight matrix
        plt.subplot(1, 3, 1)
        plt.imshow(weights_matrix[i], aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Weight Matrix of {layer_names[i]}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Input Feature Index')
        
        # subplot 2: weight distribution
        plt.subplot(1, 3, 2)
        plt.hist(weight, bins=50, alpha=0.75)
        plt.title(f'Weight Distribution of {layer_names[i]}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        
        # subplot 3: weight outlier index (max-mean)/std of each column
        plt.subplot(1, 3, 3)
        # Calculate outlier index for each column
        # Reshape the weight array to its original matrix form
        mean_per_column = np.mean(weights_matrix[i], axis=0)
        std_per_column = np.std(weights_matrix[i], axis=0)
        max_per_column = np.max(weights_matrix[i], axis=0)
        outlier_index = (max_per_column - mean_per_column) / std_per_column
        plt.plot(outlier_index)
        plt.title(f'Weight Outlier Index of {layer_names[i]}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Outlier Index')

        plt.savefig(f'weight_plot/{layer_names[i].replace("/", "_")}.png')
        # plt.show()
        plt.clf()

# Step 3: main
if __name__ == "__main__":
    plot_model_weights(model)