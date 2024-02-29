from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# OPTIONAL TO RUN ON A SPECIFIC GPU:
import os

MODEL_NAME = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Move the model to GPU and set it to half precision (float16)
model = model.half().cuda()

inp = "What color is the sky?"

input_ids = tokenizer.encode(inp, return_tensors="pt")

# Move the input to GPU  (ONLY do this if you're using the GPU for your model.)
input_ids = input_ids.cuda()

# Using automatic mixed precision
with torch.cuda.amp.autocast():
    # generate text until the output length (which includes the original input/context's length) reaches max_length
    output = model.generate(input_ids, max_length=2048, do_sample=True, early_stopping=True, num_return_sequences=1, eos_token_id=model.config.eos_token_id)

# Move the output back to CPU
output = output.cpu()
# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(output_text)