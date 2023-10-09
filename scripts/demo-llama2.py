from transformers import AutoTokenizer
import transformers
import torch

# model_path = "meta-llama/Llama-2-7b"
model_path = "daryl149/llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = """USER: 次の英文を日本語に訳してください。
I have a pen.
SYSTEM:"""

# 推論の実行
sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
print(sequences[0]["generated_text"])
