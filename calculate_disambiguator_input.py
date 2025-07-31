import ast
import pandas as pd
from typing import List
import torch
from PIL import Image
import re
import string
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from gensim.models import KeyedVectors
from pathlib import Path

device = 'cuda'

csv_path = "./flickr30k/flickr30k-descrip/flickr_annotations_30k.csv"
image_dir = Path("./flickr30k/flickr30k-images")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
w2v = KeyedVectors.load_word2vec_format(
    "./word2vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin", binary=True)

# 读取时指定分隔符，并处理空格
data = pd.read_csv(
    csv_path,
    encoding="utf-8",
    sep='|'
)

punct_re = f'[{re.escape(string.punctuation)}]'  # 转义标点符号


def word2vec(word: str) -> torch.Tensor:  # 类型提示
    if word in w2v:
        return torch.tensor(np.array(w2v[word], dtype=np.float32), dtype=torch.float, device=device)
    return torch.zeros(300, dtype=torch.float, device=device)


def get_sentence_w2v(sentence: str) -> torch.Tensor:
    word_list = sentence.split(' ')
    embeds = torch.stack([word2vec(word) for word in word_list], dim=0)
    return embeds.mean(dim=0, keepdim=False)


@torch.no_grad()
def process(sentences: List[str], file_name: str):
    # 1. 先处理空值和非字符串类型，转换为字符串
    # 将float类型的NaN转换为空字符串，其他类型强制转为字符串
    sentences = [
        "" if pd.isna(s) else str(s)
        for s in sentences
    ]

    assert len(sentences) == 5

    sentences = [re.sub(punct_re, '', s) for s in sentences]

    image_path = image_dir / file_name
    image = Image.open(image_path)

    inputs = processor(
        text=sentences,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    image_embeds = outputs.image_embeds.view(-1).cpu().tolist()
    text_embeds = outputs.text_embeds.view(-1).cpu().tolist()

    w2v_embeds = torch.stack([get_sentence_w2v(s) for s in sentences], dim=0).view(-1).cpu().tolist()

    return pd.Series(
        {"image_embeds": image_embeds,
         "text_embeds": text_embeds,
         "labels": w2v_embeds
         }
    )


# 对原始数据每一行使用process函数处理
from tqdm import tqdm

processed_data = []
# 初始化索引
i = 0
# 获取数据总行数
total_rows = len(data)
# 创建tqdm进度条
with tqdm(total=total_rows, desc="Processing data") as pbar:
    while i < total_rows:
        # 获取当前行的file_name
        file_name = data.iloc[i]["image_name"]
        # 初始化当前file_name对应的sentences列表
        sentences = [data.iloc[i]["comment,,,,,,,,,"]]

        # 移动到下一行
        i += 1
        pbar.update(1)

        # 检查后续行是否有相同的file_name
        while i < total_rows and data.iloc[i]["image_name"] == file_name:
            # 将句子添加到当前sentences列表
            sentences.append(data.iloc[i]["comment,,,,,,,,,"])
            i += 1
            pbar.update(1)

        # 使用process函数处理
        result = process(sentences, file_name)
        processed_data.append(result)
    # except Exception as e:
    #     print(f"Error processing row {idx}: {e}")
    #     continue

# 将处理结果转换为DataFrame并保存
result_df = pd.DataFrame(processed_data)