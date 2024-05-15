from transformers import pipeline
import torch
import pickle

original_texts = "txts/gt.txt"
erroneous_texts = "txts/error.txt"
output_file = "outs/8b-res.pkl"

access_token = "**access token**"

pipe = pipeline("text-generation",
        token = access_token,
        model = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
        torch_dtype = torch.float16,
        device_map = "auto")

with open(original_texts, 'r') as f:
    gts = f.readlines()
with open(erroneous_texts, 'r') as f:
    errors = f.readlines()
responses = []

for i in range(0,90):
    text1 = gts[i]
    text2 = errors[i]
    prompt = f"<s>USER: 你好，請只根據短文一的內容，判斷短文二的正確性。若短文二有誤，請列出違背事實或無法驗證之處。請遵循1. 錯誤一\n 2. 錯誤二\n 的格式直接回應。以下是短文一 : 「{text1}」以下是短文二 : 「{text2}」ASSISTANT:"
    out = pipe(prompt, max_new_tokens=512)
    response = out[0]["generated_text"].split("ASSISTANT:")[1]
    responses.append(response)

with open(output_file, "wb") as f:
    pickle.dump(responses, f)
