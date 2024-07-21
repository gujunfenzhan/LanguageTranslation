import argparse
import os.path
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utils import TranslationUtils
import tqdm
from torch import nn, optim, log_softmax

from dataset import TranslationDataset, TranslationDataLoader
import torch.cuda
from tokenizers import Tokenizer
from torchtext.vocab import build_vocab_from_iterator
from model import TranslationTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="en_to_zh", type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--save_interval', default=500, type=int)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--d_model', default=512, type=int)
args = parser.parse_args()
train_path = Path("data/%s" % args.dataset)
train_file_en = train_path / "train.en"
train_file_zh = train_path / "train.zh"
vocab_en_path = train_path / "vocab_en.pt"
vocab_zh_path = train_path / "vocab_zh.pt"
index_en_path = train_path / "index_en.txt"
index_zh_path = train_path / "index_zh.txt"
if torch.cuda.is_available():
    print("启用CUDA")
    device = torch.device("cuda")
else:
    print("启用CPU")
    device = torch.device("cpu")
# 读取文件内容
zh_sentences = []
en_sentences = []

tokenizer = Tokenizer.from_file('./bert-base-uncased/tokenizer.json')
# 计算文件行数
print("计算文件行数...")
with open(train_file_zh, encoding='utf-8', mode='r') as file:
    file_length = sum(1 for _ in file)
print("文件行数:%s" % file_length)

# 构建读取索引
if not index_en_path.exists():
    print("构建英文读取索引表...")
    TranslationUtils.gen_index_table(index_en_path, train_file_en, file_length, desc="英文索引表")

if not index_zh_path.exists():
    print("构建中文读取索引表...")
    TranslationUtils.gen_index_table(index_zh_path, train_file_zh, file_length, desc="中文索引表")

print("加载索引表...")
with open(index_en_path, encoding="utf-8", mode='r') as file:
    index_en_mapping = list(file.readlines())
with open(index_zh_path, encoding="utf-8", mode='r') as file:
    index_zh_mapping = list(file.readlines())

# 构建词汇表
if not vocab_en_path.exists():
    print("构建英文词汇表...")
    with tqdm.tqdm(open(train_file_en, encoding="utf-8", mode="r"), total=file_length, desc="英文词汇表") as file:
        vocab_en = build_vocab_from_iterator((tokenizer.encode(line, add_special_tokens=False).tokens for line in file),
                                             min_freq=2, specials=["<s>", "</s>", "<pad>", "<unk>"])
    vocab_en.set_default_index(vocab_en["<unk>"])
    torch.save(vocab_en, vocab_en_path)
else:
    vocab_en = torch.load(vocab_en_path, map_location="cpu")

if not vocab_zh_path.exists():
    print("构建中文词汇表...")
    with tqdm.tqdm(open(train_file_zh, encoding="utf-8", mode="r"), total=file_length, desc="中文词汇表") as file:
        vocab_zh = build_vocab_from_iterator((list(line.strip().replace("", "")) for line in file),
                                             min_freq=1, specials=["<s>", "</s>", "<pad>", "<unk>"])
    vocab_zh.set_default_index(vocab_zh["<unk>"])
    torch.save(vocab_zh, vocab_zh_path)
else:
    vocab_zh = torch.load(vocab_zh_path, map_location="cpu")

train_dataset = TranslationDataset(train_file_en, train_file_zh,
                                   lambda i: tokenizer.encode(i, add_special_tokens=False).tokens,
                                   lambda i: list(i.strip().replace("", "")),
                                   vocab_en, vocab_zh, index_en_mapping, index_zh_mapping)
# train_dataloader = TranslationDataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size, max_sentence_length=args.max_length, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=64)
train_dataloader = TranslationDataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size,
                                         max_sentence_length=args.max_length)

if Path('model.pth').exists():
    model = torch.load('model.pth')
else:
    model = TranslationTransformer(args.d_model, vocab_en, vocab_zh, args.max_length, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.num_epochs):
    tq = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for idx, item in tq:
        src, tgt, y, n_tokens = item
        src, tgt, y = src.to(device), tgt.to(device), y.to(device)
        y_p = model(src, tgt)
        y_p = model.predictor(y_p)
        shape = y_p.shape
        y_p = y_p.reshape((shape[0] * shape[1], -1))
        y = y.reshape((shape[0] * shape[1],))
        loss = criterion(y_p, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tq.set_postfix(loss=loss.item())
        if (idx + 1) % args.save_interval == 0:
            torch.save(model, "model.pth")
    torch.save(model, "model.pth")
