"""
Supervised Fine-Tuning (SFT) with QLoRA

Pretrained Model: Llama 2 7B

Reference: https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32

"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import time

# %% Parameters

# The model that you want to train from the Hugging Face hub
model_name = "/home/will/Documents/models/Llama-2-7b-chat-hf"  # "meta-llama/Llama-2-7b-chat-hf"

# Fine-tuned model name
new_model = "llama-2-7b-Guanaco-QLoRA-poc"

# Merged new model
new_model_full = "llama-2-7b-Guanaco-QLoRA-poc_full"

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}
# %%

""" 
#===============================================================================
# Case 1 : Run Llama 2 7B as is
#===============================================================================
"""
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,  # torch.float32 requires more than 24G of vram
    device_map=device_map,
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

################################################################################
# Test it out
################################################################################

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
start_time = time.time()
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(f"Generate time: {time.time() - start_time:.2f} seconds")  # 3.9 seconds
print(result[0]['generated_text'])


# %%

""" 
#===============================================================================
# Case 2: Load Base Model then Merge with Fine Tuned Checkpoint
#===============================================================================
"""

################################################################################
# load base model and tokenizers
################################################################################



# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()


################################################################################
# Test it out
################################################################################

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
start_time = time.time()
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(f"Generate time: {time.time() - start_time:.2f} seconds")  # 3.6 seconds
print(result[0]['generated_text'])



# %%

""" 
#===============================================================================
# Case 3: Load Merged Model (Base + Fine Tuned) directly and Run with 4 bit quantization
#===============================================================================
"""
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


# Load base model
model = AutoModelForCausalLM.from_pretrained(
    new_model_full,
    quantization_config=bnb_config,
    device_map=device_map
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

################################################################################
# Test it out
################################################################################

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
start_time = time.time()
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(f"Generate time: {time.time() - start_time:.2f} seconds")  # 2.6 seconds
print(result[0]['generated_text'])



