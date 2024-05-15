from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle

original_texts = "txts/gt.txt"
erroneous_texts = "txts/error.txt"
output_file = "outs/breeze-res.pkl"

# Instruction Model
model = AutoModelForCausalLM.from_pretrained(
    "MediaTek-Research/Breeze-7B-Instruct-v1_0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2" # optional
)

tokenizer = AutoTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v1_0")

with open(original_texts, 'r') as f:
    gts = f.readlines()
with open(erroneous_texts, 'r') as f:
    errors = f.readlines()
responses = []

for i in range(0,90):
    text1 = gts[i]
    text2 = errors[i]
    chat = [
      {"role": "user", "content": "你好，請只根據短文一的內容，判斷短文二的正確性。若短文二有誤，請列出違背事實或無法驗證之處。請遵循1. 錯誤一\n 2. 錯誤二\n 的格式直接回應。"},
      {"role": "assistant", "content": "沒問題！請提供短文一與短文二給我，來幫助您判斷。"},
      {"role": "user", "content": f"太棒了！以下是短文一 : 「{text1}」以下是短文二 : 「{text2}」"},
    ]

    outputs = model.generate(tokenizer.apply_chat_template(chat, return_tensors="pt"),
                             # adjust below parameters if necessary
                             max_new_tokens=128,
                             top_p=0.01,
                             top_k=85,
                             repetition_penalty=1.1,
                             temperature=0.01)

    outtext = tokenizer.decode(outputs[0])
    response = outtext.split("[/INST]")[-1][1:-4] # get rid of preceeding text and thhe trailing </s>
    responses.append(response)

with open(output_file, "wb") as f:
    pickle.dump(responses, f)
