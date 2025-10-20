### Script to run ***hellaswag*** through lm-eval-harness

```sh
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 1 \
    --write_out
```
