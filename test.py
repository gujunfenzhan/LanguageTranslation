from pathlib import Path

import torch
from tokenizers import Tokenizer

train_path = Path("data/%s" % "en_to_zh")
train_file_en = train_path / "train.en"
train_file_zh = train_path / "train.zh"
vocab_en_path = train_path / "vocab_en.pt"
vocab_zh_path = train_path / "vocab_zh.pt"
model = torch.load('model.pth')
vocab_en = torch.load(vocab_en_path, map_location="cpu")
vocab_zh = torch.load(vocab_zh_path, map_location="cpu")
# 加载基础的分词器模型，使用的是基础的bert模型。`uncased`意思是不区分大小写
tokenizer = Tokenizer.from_file('./bert-base-uncased/tokenizer.json')


def en_tokenizer(line):
    """
    定义英文分词器，后续也要使用
    :param line: 一句英文句子，例如"I'm learning Deep learning."
    :return: subword分词后的记过，例如：['i', "'", 'm', 'learning', 'deep', 'learning', '.']
    """
    # 使用bert进行分词，并获取tokens。add_special_tokens是指不要在结果中增加‘<bos>’和`<eos>`等特殊字符
    return tokenizer.encode(line, add_special_tokens=False).tokens


device = torch.device('cuda')


def translate(src: str):
    """
    :param src: 英文句子，例如 "I like machine learning."
    :return: 翻译后的句子，例如：”我喜欢机器学习“
    """

    # 将与原句子分词后，通过词典转为index，然后增加<bos>和<eos>
    src = torch.tensor([0] + vocab_en(en_tokenizer(src)) + [1]).unsqueeze(0).to(device)
    # 首次tgt为<bos>
    tgt = torch.tensor([[0]]).to(device)
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(128):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # 如果为<eos>，说明预测结束，跳出循环
        if y == 1:
            break
    # 将预测tokens拼起来
    tgt = ''.join(vocab_zh.lookup_tokens(tgt.squeeze().tolist()))
    return tgt


while True:
    en_sentence = input("英文:")
    out = translate(en_sentence)
    print(out)
    print(len(out))
