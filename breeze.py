from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import load_dataset
import torch
import pickle

huggingface_token = "**access token**"
dataset_type = "short" # or "medium" or "long"
dataset = load_dataset("json", data_files=f"cjli/finegrained-halu-tw/dataset_{dataset_type}.jsonl", use_auth_token=huggingface_token)

output_file = "outs/breeze-res.pkl"

# Instruction Model
model = AutoModelForCausalLM.from_pretrained(
    "MediaTek-Research/Breeze-7B-Instruct-v1_0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2" # optional
)

tokenizer = AutoTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v1_0")

responses = []
for data in dataset:
    text1 = data["ground_truth"]
    text2 = data["content_w_errors"]
    chat = [
      {"role": "user", "content": "你好，請只根據短文一的內容，判斷短文二的正確性。若短文二有誤，請列出違背事實或無法驗證之處。請遵循1. 錯誤一\n 2. 錯誤二\n 的格式直接回應。"},
      {"role": "assistant", "content": "沒問題！請提供短文一與短文二給我，來幫助您判斷。"},
      {"role": "user", "content": f"太棒了！以下是短文一 : 「{text1}」以下是短文二 : 「{text2}」"},
    ]

    outputs = model.generate(tokenizer.apply_chat_template(chat, return_tensors="pt"),
                             # adjust below parameters if necessary
                             max_new_tokens=512,
                             top_p=0.01,
                             top_k=85,
                             repetition_penalty=1.1,
                             temperature=0.01)

    outtext = tokenizer.decode(outputs[0])
    response = outtext.split("[/INST]")[-1][1:-4] # get rid of preceeding text and thhe trailing </s>
    responses.append(response)

with open(output_file, "wb") as f:
    pickle.dump(responses, f)
