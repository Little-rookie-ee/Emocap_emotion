# -*- coding: utf-8 -*-
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Processor, HubertModel
import librosa
# import soundfile as sf
import random
'''
* 用于整合文本数据，音频特征、标签
'''
device = 'cpu' if torch.cuda.is_available() else 'cpu'

import logging

## 创建日志输出到文件和操作台
def creat_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formater = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ## 创建handler将日志输出到文档
    file_handler = logging.FileHandler(filename='./log/data_process_log.log')
    file_handler.setFormatter(formater)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    ## 创建handler将日志输出到操作台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formater)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    return logger

## 结构化封装数据，文本、音频序列、标签
class text_audio_label(object):
    def __init__(self, text, audio, label):
        self.text = text
        self.audio = audio
        self.label = label

## 提取音频的特征
def audio_feature(audio, rate):

    processor = Wav2Vec2Processor.from_pretrained("./pretrained_model/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("./pretrained_model/wav2vec2-base-960h")

    model.to(device)
    input = processor(audio, sampling_rate=rate, return_tensors="pt")
    print(input['input_values'].shape)
    input.to(device)
    out = model(**input).last_hidden_state
    print(out)
    return out[0]

#读取数据
def data_processing(path, logger):
    data = []
    emotions = os.listdir(path)

    l = 0  ## labels
    for emotion in emotions:
        print('----------{}------------'.format(emotion))
        t = 0 ## 统计真正有的数据量
        k = 0 ## 统计实际用的数据量
        path_text = (path+'/'+emotion+'/'+'words')
        path_audio = (path+'/'+emotion+'/'+ 'wavs')
        sents = os.listdir(path_text)
        for sent in sents:
            #print('------{}------'.format(t))
            z = os.path.getsize(path_text+'/'+sent)
            t += 1
            if z != 0:
                k += 1
                #qq += 1
                r = open(path_text+'/'+sent)
                text = r.read()
                audio, rate = librosa.load((path_audio+'/'+sent).replace('.txt', ''), sr=16000)
                data.append(text_audio_label(text=text, audio=audio, label=l))
                #texts.append(text)
                #audio_len += len(audio)
                #audios.append(audio)

        l += 1
        logger.info('emotion of {} have {} true data, and use {} data'.format(emotion, t, k))

    return data

if __name__=="__main__":
    logger = creat_logger()
    data = data_processing('./data', logger)
    """audio, rate = librosa.load('/workspace/practice/双模（语音文本）/IEMOCAP_Emotion/data/anger/wavs/Ses01F_impro01_F012.wav', sr=16000)
    # a, r = sf.read('F:\python_data\practice\双模（语音文本）\IECOMAP_Emotion\data\excitement_happy\wavs\Ses01F_impro03_F000.wav')
    print(audio)
    processor = Wav2Vec2Processor.from_pretrained("./pretrained_model/wav2vec2-base-960h")
    k = processor(audio, sampling_rate=rate, return_tensors='pt')
    hubert = HubertModel.from_pretrained('/workspace/practice/双模（语音文本）/IEMOCAP_Emotion/pretrained_model/hubert-base-ls960')
    out = hubert(**k).last_hidden_state[0]
    print(k)
    print(out.shape)"""

    # print(a)