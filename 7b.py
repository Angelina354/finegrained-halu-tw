from transformers import pipeline
from dataset import load_dataset
import torch
import pickle

huggingface_token = "**access token**"
dataset_type = "short" # or "medium" or "long"
dataset = load_dataset("json", data_files=f"cjli/finegrained-halu-tw/dataset_{dataset_type}.jsonl", use_auth_token=huggingface_token)

output_file = "outs/7b-res.pkl"

taide_access_token = "**access token**"
pipe = pipeline("text-generation",
        token = taide_access_token,
        model = "taide/TAIDE-LX-7B-Chat",
        torch_dtype = torch.float16,
        device_map = "auto")

responses = []
for data in dataset:
    text1 = data["ground_truth"]
    text2 = data["content_w_errors"]
    prompt = f"<s>USER: 你好，請只根據短文一的內容，判斷短文二的正確性。若短文二有誤，請列出違背事實或無法驗證之處。請遵循1. 錯誤一\n 2. 錯誤二\n 的格式直接回應。以下是短文一 : 「{text1}」以下是短文二 : 「{text2}」ASSISTANT:"
    out = pipe(prompt, max_new_tokens=512)
    response = out[0]["generated_text"].split("ASSISTANT:")[1]
    responses.append(response)

with open(output_file, "wb") as f:
    pickle.dump(responses, f)
