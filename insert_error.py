from transformers import pipeline
import torch

infile = "unknown.txt"
outfile = "out-unknown.txt"

pipe = pipeline("text-generation",
        model = "/home/u4249612/llama2/",
        torch_dtype = torch.float16,
        device_map = "auto")

op = "請根據下面這則短文的內容，以專家或熟悉人士的角度，想像出其中可能的缺失以及補強方法 :「"
with open(infile, 'r') as f:
    articles = f.readlines()

cleaned = []
for opp in articles:
    text = "<s>USER:" + op + opp + "」ASSISTANT:"
    out = pipe(text)
    article = out[0]["generated_text"].split("ASSISTANT:")[1]
    with open(outfile, "a") as f:
        f.write(article)
        f.write("\n= = = = = = = = = = = = = = =\n")
