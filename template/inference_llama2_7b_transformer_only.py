"""
Llama 2 Inference with only transformer and pytorch

Pretrained Model: Llama 2 7B

"""
import time
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

model_id = "/home/will/Documents/models/Llama-2-7b-chat-hf"

# bnb_config = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=True)
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", offload_folder="save_folder")

tokenizer = AutoTokenizer.from_pretrained(model_id)


prompt = input("Enter your question ('quit' to quit): ")
prompt_template=f'''### Human: {prompt}
### Assistant:
'''

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)


start_time = time.time()
print(pipe(prompt_template)[0]['generated_text'])
print(f"Generate time: {time.time() - start_time:.2f} seconds")

