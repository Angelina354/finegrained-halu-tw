#!/usr/bin/env python
from transforemrs import pipeline
import torch
#import pickle # not nessary

def load_file(file: str = None):
    if file:
        with open(file, 'r') as f:
            return f.readlines()

def main(
    model_id: str,
    output: str,
    input_origin: str="txts/gt.txt",
    input_errors: str="txts/error.txt",
):
    generator = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pass

if __name__ == "__main__":
    import fire 
    fire.Fire(main)