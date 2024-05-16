## Fine-Grained Halu Taiwan

### Dataset
- `dataset.jsonl`
- 欄位 : id, ground_truth (原始短文), content_w_errors (加完錯誤的), type (預設的插入錯誤類別), errors (文中錯誤), errors_type (文中各錯誤類別)

### 錯誤產生過程 (已完成)

1. `txts/gt.txt` 為整理好的短文，每篇一行
2. 更改 `insert_error.py` 的輸入輸出檔名及產生錯誤的 prompt
3. 整理結果和錯誤 (目前存在 google sheet，底下的 txt 都從那邊複製貼過來)

### 交給模型偵錯

1. 原始的短文 (`txts/gt.txt`) 與加完錯誤的短文 (`txts/error.txt`) 每篇一行儲存
2. 直接跑各模型對應的 `{modelname}.py`，依所需更改輸入輸出檔名和 access token
3. 輸出存於 `outs/{modelname}-res.pkl`，格式為 `[{第一篇回應}, {第二篇回應}, ...]`

### 評估模型表現 (gpt-4)

1. 使用加完錯誤的短文 (`txts/error.txt`) 與所有錯誤列表 (`txts/wronglist.txt`)，後者為每個錯誤一行，每題分別編號為 1. 至 i.

2. `python gpt4-judge.py {modelname}`  (7b, 8b or breeze)

3. 輸出存於 `outs/eval-{modelname}.txt`，格式如下
  ```
  [Label]
  Yes/No
  [Label]
  Yes/No
  ...
  
  =====
  (下一篇的判斷)
  ```

### 計算分數

1. 使用評估結果 (`outs/eval-{modelname}.txt`) 與錯誤類別列表 (`txts/wrongtype.txt`)，後者分別對應到所有錯誤列表各行的錯誤類別，六種錯誤由 1 至 6 編號
2. `python score.py {modelname}`  (7b, 8b or breeze)
3. 直接印出每類錯誤 & 整體的正確率
