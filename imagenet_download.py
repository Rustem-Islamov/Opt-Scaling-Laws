import os
from datasets import load_dataset

print("HOME =", os.environ.get("HOME"))
print("HF_HOME =", os.environ.get("HF_HOME"))

ds = load_dataset("ILSVRC/imagenet-1k", split="test", cache_dir="/scratch/$USER/huggingface/datasets")
print(len(ds))
print("Cache files:", ds.cache_files)