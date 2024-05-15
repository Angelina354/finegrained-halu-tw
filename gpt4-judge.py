from openai import OpenAI
import pickle
import sys

client = OpenAI(api_key = "**api key**")
system_content = "您精通繁體中文書寫，且善於推理，明察秋毫。"

modelname = sys.argv[1] # 7b, 8b or breeze
erroneous_texts = "txts/error.txt"
list_of_all_errors = "txts/wronglist.txt"
outfile = f"outs/eval-{modelname}.txt"

with open(f'outs/{modelname}-res.pkl', 'rb') as f:
    model_anss = pickle.load(f)
with open(erroneous_texts, 'r') as f:
    error_texts = f.readlines()
with open(list_of_all_errors, 'r') as f:
    error_list = f.readlines()

cnt = 0
for i in range(0,90):
    print(i)
    text = error_texts[i].strip()
    model_ans = model_anss[i].strip()
    #print(model_ans)
    while True:
        error = error_list[cnt].strip()
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
        #exit(0)
        completion = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": judge_prompt},
            ],
        )

        judge_result = completion.choices[0].message.content
        #print(judge_result)
        
        with open(outfile, "a") as f:
            f.write(judge_result)
            f.write("\n")
        cnt += 1
        if (error_list[cnt][0:2])=='1.': break

    with open(outfile, "a") as f:
        f.write("\n=====\n")
