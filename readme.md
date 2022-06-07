## 双模情感识别（语音、文本）

本仓库使用做的是一个基于**语音**与**文本**的双模情感识别。

### 数据集

IEMOCAP

对情感进行四分类，anger、excitement_happy、neutral、sadness

文本由语音数据通过百度语音识别api转录实现，其中含有部分结果识别空白，将其剔除。

各类数据实际使用数量：

平均长度：74431

| Emotions         | Reality | Used |
|:----------------:|:-------:|:----:|
| Anger            | 1103    | 1095 |
| Excitement_happy | 1636    | 1589 |
| Neutral          | 1708    | 1632 |
| Sadness          | 1084    | 989  |
| **total**        | 5531    | 5305 |

### 结果

| Text/model          | Audio/model | Audio feature | fusion    | Accuracy    |
|:-------------------:|:-----------:|:-------------:| --------- |:-----------:|
| BERT+LSTM           |             |               | concat    | 63.6%       |
| BERT+LSTM+Attention |             |               | concat    | 64.2%       |
|                     | BILSTM      | mfcc          | concat    | 53.1%       |
|                     | LSTM        | hubert        | concat    | 61.7%       |
| BERT+LSTM           | LSTM        | mfcc          | concat    | 64.2%       |
| BERT+LSTM           | LSTM        | hubert        | concat    | 70.9%，69.5% |
| BERT+LSTM           | LSTM        | hubert        | attention | 71.7%       |

双模态，文本使用模型：**BERT+LSTM**，音频使用模型：**LSTM**

| 音频特征 | accuracy |     |
| ---- | -------- | --- |
| mfcc |          |     |
|      |          |     |
|      |          |     |

### 遇到的问题

1. **问题描述**：对于音频进行padding时，对于音频序列信号（librosa读取）paddding后再获取MFCC特征运行正常，但是先获取音频MFCC再padding，会导致训练时各个batch中不同数据LSTM的输出保持一致，因此导致训练精度无法提升
   
   **尝试方法**：
   
   1. 调整学习率（考虑可能是过大，导致参数变得太大，导致输入对结果影响不大），无效
   
   2. 增加BN（考虑可能是输入的数据跨度太大），无效
