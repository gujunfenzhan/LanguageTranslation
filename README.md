# 基于transformer架构实现英译中

## 文件结构

+ data 数据集
  + dataset_name 数据集名字
    + train.en 英文资料(与中文一对一)
    + train.zh 中文资料(与英文一对一)

## 训练

```bash
python train.py
--dataset         数据集
--lr              学习率
--batch_size      训练批次大小
--num_epochs      训练次数
--save_interval   保存间隔, 以批次为单位
--max_length      训练时送入模型向量长度
--d_model         词嵌入输出维度
```

## 测试

```bash
python test.py
```

## 效果

训练集共计1万条，只训练了一次，继续train应该效果可以更好。

```text
英文:Okay, this is a very interesting project.
项目翻译:好吧，这是个非常有趣的项目
百度翻译:好吧，这是一个非常有趣的项目。

英文:But if that strategy is conceived as mastering two dynamic processes, overcoming the ecological constraint could be an accelerator of growth.
项目翻译:但如果这种策略被认为是两种动态的过程，那么，克服生态限制可能是经济增长的一个加速。
谷歌翻译:但是，如果将这一战略视为掌握两个动态过程，那么克服生态限制可能是增长的加速器。

英文:Health ministers will not be able to cope with an increase in infectious diseases due to global warming.
项目翻译:卫生部长不能因全球变暖而加剧传染病的增长。
谷歌翻译:由于全球变暖，卫生部长们将无法应对传染病的增加。
```

## 参考文章
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Transformers 库快速入门教程 Transformer基础理论以及Encoder代码编写](https://transformers.run/)

[CSDN 采用pytorch预设的Transformer训练英译中](https://blog.csdn.net/zhaohongfei_358/article/details/126175328)

[周弈帆 PyTorch Transformer 英中翻译超详细教程](https://zhouyifan.net/2023/06/11/20221106-transformer-pytorch/)
