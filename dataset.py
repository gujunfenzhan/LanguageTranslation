import os
import sys
from multiprocessing import Lock

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(self,train_en_path, train_zh_path,tokenizer_en, tokenizer_zh, vocab_en, vocab_zh, index_en_mapping, index_zh_mapping):
        self.en_to_zh_mappings = []
        self.tokenizer_en = tokenizer_en
        self.tokenizer_zh = tokenizer_zh
        self.vocab_en = vocab_en
        self.vocab_zh = vocab_zh
        self.train_en_io = open(train_en_path, encoding="utf-8", mode='r')
        self.train_zh_io = open(train_zh_path, encoding="utf-8", mode='r')
        for idx in range(len(index_en_mapping)):
            self.en_to_zh_mappings.append((int(index_en_mapping[idx].strip()), int(index_zh_mapping[idx].strip())))
        self.lock = Lock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.train_en_io.close()
        self.train_zh_io.close()

    def __len__(self):
        return len(self.en_to_zh_mappings)

    def __getitem__(self, idx):
        src_idx, target_idx = self.en_to_zh_mappings[idx]
        # 读取内容
        with self.lock:
            self.train_en_io.seek(src_idx)
            src = self.train_en_io.readline()
            self.train_zh_io.seek(target_idx)
            target = self.train_zh_io.readline()

        # 分词
        src, target = self.tokenizer_en(src), self.tokenizer_zh(target)
        # 映射
        src, target = self.vocab_en(src), self.vocab_zh(target)

        return src, target

class TranslationDataLoader(DataLoader):
    def __init__(self, max_sentence_length=72, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sentence_length = max_sentence_length

        self.collate_fn = self.custom_collate_fn

    def custom_collate_fn(self, bath):
        bos_id = torch.tensor([0])
        eos_id = torch.tensor([1])
        pad_id = 2

        src_list, tgt_list = [], []

        for _src, _tgt in bath:
            processed_src = torch.cat(
                (
                    bos_id,
                    torch.tensor(
                        _src,
                        dtype=torch.int64
                    ),
                    eos_id
                ),
                dim=0
            )
            processed_tgt = torch.cat(
                (
                    bos_id,
                    torch.tensor(
                        _tgt,
                        dtype=torch.int64
                    ),
                    eos_id
                ),
                dim=0
            )
            # 长度不足填充
            src_list.append(pad(processed_src, (0, self.max_sentence_length-len(processed_src)), value=pad_id))
            tgt_list.append(pad(processed_tgt, (0, self.max_sentence_length-len(processed_tgt)), value=pad_id))
        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        # 预测真实值
        y = tgt[:, 1:]
        # 输入decoder
        tgt = tgt[:, :-1]
        n_tokens = (y != 2).sum()
        return src, tgt, y, n_tokens
