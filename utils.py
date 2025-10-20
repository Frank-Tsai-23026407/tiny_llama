from transformers import StoppingCriteria
import torch 

def input_formatting(history, input_text):
    history_transformer_format = history + [[input_text, ""]]
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    return messages

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1] == 2 # EOS token id for TinyLlama
