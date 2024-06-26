from openai import OpenAI
from dataset import load_dataset
from tqdm import tqdm
import pickle, sys

huggingface_token = "**access token**"
dataset_type = "short" # or "medium" or "long"
dataset = load_dataset("json", data_files=f"cjli/finegrained-halu-tw/dataset_{dataset_type}.jsonl", use_auth_token=huggingface_token)

client = OpenAI(api_key = "**api key**")
system_content = "您精通繁體中文書寫，且善於推理，明察秋毫。"

modelname = sys.argv[1] # 7b, 8b or breeze
outfile = f"outs/eval-{modelname}.txt"

with open(f'outs/{modelname}-res.pkl', 'rb') as f:
    model_anss = pickle.load(f)

for i, data in enum(tqdm(dataset)):
    text = data["content_w_errors"]
    errors = data["errors"]
    model_ans = model_anss[i].strip()
    for error in errors:
        judge_prompt = f"""以下是一篇短文[Text]和它裡面出現的錯誤資訊[Error]。您的任務是判斷模型回答[Model Answer]有無找出這項錯誤，有找出錯誤為Yes，無找出錯誤為No。

[Text]
{text}
[Text]

[Error]
{error}
[Error]

[Model Answer]
{model_ans}
[Model Answer]

請嚴格遵循以下輸出格式，輸出判斷標籤[Label]，並確保[Label]的回答僅包含[Yes]或[No]。請注意，回答應直接明瞭，避免額外的字元及數字。輸出格式如下：
[Label]
判斷標籤 (Yes/No)
"""
        """
        請嚴格遵循以下輸出格式，先輸出判斷理由[Reason]，再輸出判斷標籤[Label]，並確保[Label]的回答僅包含[Yes]或[No]。請注意，回答應直接明瞭，避免額外的字元及數字。輸出格式如下：
[Reason]
判斷理由
[Reason]
[Label]
判斷標籤 (Yes/No)
[Label]
        """
        #print(judge_prompt)
        
        completion = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": judge_prompt},
            ],
        )

        judge_result = completion.choices[0].message.content
        
        with open(outfile, "a") as f: # finish judging one error
            f.write(judge_result)
            f.write("\n")

    with open(outfile, "a") as f: # finish judging one article
        f.write("\n=====\n")
