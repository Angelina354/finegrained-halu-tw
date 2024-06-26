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
outfile = f"outs/eval-{modelname}.txt" # save GPT-4's judging result

with open(f'outs/{modelname}-res.pkl', 'rb') as f:
    model_anss = pickle.load(f)

result = [[], [], [], [], [], []]
for i, data in enum(tqdm(dataset)):
    text = data["content_w_errors"]
    errors = data["errors"]
    errortypes = data["errors_type"]
    model_ans = model_anss[i].strip()
    for i, error in enum(errors):
        errortype = int(errortypes[i]) # the subtype of this error
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
        #print(judge_prompt)
        
        completion = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": judge_prompt},
            ],
        )
        judge_result = completion.choices[0].message.content
        
        if "Yes" in judge_result: result[errortype-1].append(1)    # error found
        else: result[errortype-1].append(0)                        # error missed

        with open(outfile, "a") as f:                              # finish judging one error
            f.write(judge_result)
            f.write("\n")

    with open(outfile, "a") as f:                                  # finish judging one article
        f.write("\n=====\n")


# Get accuracy
# errorcode: noun 1, relation 2, sentence 3 -> factual errors
#           imagined 4, comment 5, unknown 6 -> unverifiable errors
fact = result[0]+result[1]+result[2]
unve = result[3]+result[4]+result[5]
print(f"---Result of model {modelname}---")
print("Total acc:", sum(fact+unve), '/', len(fact+unve), ';', sum(fact+unve)/len(fact+unve))
print("Fact. err:", sum(fact), '/', len(fact), ';', sum(fact)/len(fact))
print("Unvr. err:", sum(unve), '/', len(unve), ';', sum(unve)/len(unve))

print("---Error subtype accuracies---")
print("1- noun acc:", sum(result[0]), '/', len(result[0]), ';', sum(result[0])/len(result[0]))
print("2- rela acc:", sum(result[1]), '/', len(result[1]), ';', sum(result[1])/len(result[1]))
print("3- sent acc:", sum(result[2]), '/', len(result[2]), ';', sum(result[2])/len(result[2]))
print("4- imag acc:", sum(result[3]), '/', len(result[3]), ';', sum(result[3])/len(result[3]))
print("5- comm acc:", sum(result[4]), '/', len(result[4]), ';', sum(result[4])/len(result[4]))
print("6- unkn acc:", sum(result[5]), '/', len(result[5]), ';', sum(result[5])/len(result[5]))
