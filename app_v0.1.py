import torch
from peft import PeftModel
import transformers
import gradio as gr
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import Trainer

BASE_MODEL = "TheBloke/vicuna-7B-1.1-HF"

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map = "auto",
    offload_folder="./cache",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def format_prompt(prompt: str) -> str:
    return f"### Human: {prompt}\n### Assistant:"

generation_config = GenerationConfig(
    max_new_tokens=128,
    temperature=0.2,
    repetition_penalty=1.0,
)

def generate_text(prompt: str):
    formatted_prompt = format_prompt(prompt)

    inputs = tokenizer(
        formatted_prompt, 
        padding=False, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        tokens = model.generate(**inputs, generation_config=generation_config)

    response = tokenizer.decode(tokens[0], skip_special_tokens=True)
    assistant_index = response.find("### Assistant:") + len("### Assistant:")
    return response[assistant_index:].strip()

iface = gr.Interface(
    fn=generate_text, 
    inputs="text", 
    outputs="text",
    title="Chatbot",
    description="This vicuna app is using this model: https://huggingface.co/TheBloke/vicuna-7B-1.1-HF"
)
iface.launch()
