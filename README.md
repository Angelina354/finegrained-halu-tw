## Fine-Grained Halu Taiwan

### Step 0. Dataset (放在 HuggingFace)

### Step 1. 交給模型偵錯

1. 可參考各模型對應的 `{modelname}.py`，依所需更改輸入輸出檔名和 HuggingFace 資料 & 模型的 access token
2. 輸出存於 `outs/{modelname}-res.pkl`，格式為 `[{第一篇回應}, {第二篇回應}, ...]`

### Step 2. 評估模型表現 (gpt-4)

1. `python gpt4-judge.py {modelname}`  (7b, 8b or breeze)
2. 輸出存於 `outs/eval-{modelname}.txt`，格式如下
  ```
  [Label]
  Yes/No
  [Label]
  Yes/No
  ...
  
  =====
  (下一篇的判斷)
  ```
3. 也會直接算出每類錯誤 & 整體的正確率

### Backup. 計算分數

1. 可單就先前儲存的 GPT-4 輸出結果 (`outs/eval-{modelname}.txt`) 重新算分
2. `python score.py {modelname}`  (7b, 8b or breeze)
