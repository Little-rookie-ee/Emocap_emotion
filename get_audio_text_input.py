from transformers import BertTokenizer, Wav2Vec2Processor, Wav2Vec2Model, HubertModel
from args_setting import args
import torch
import librosa
import numpy as np


"""
** 获取音频、文本的各个特征输入
"""


def text_bert(text):
    """text_ids = []
    text_attention_mask = []
    text_token_type_ids = []"""
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)  ## 创建bert的输入
    # for (idx, example) in enumerate(data):
    text_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=args.text_max_len,
        padding='max_length',
        truncation=True  ## 保证超过max_length的句子自动截断
    )
    assert len(text_input['input_ids']) == args.text_max_len, "Error with text input length {} vs {}".format(
        len(text_input['input_ids']), args.text_max_len)
    assert len(text_input['attention_mask']) == args.text_max_len, "Error with text input length {} vs {}".format(
        len(text_input['attention_mask']), args.text_max_len)
    assert len(text_input['token_type_ids']) == args.text_max_len, "Error with text input length {} vs {}".format(
        len(text_input['token_type_ids']), args.text_max_len)
    """text_ids.append(text_input['input_ids'])
        text_attention_mask.append(text_input['attention_mask'])
        text_token_type_ids.append(text_input['token_type_ids'])"""
    # return text_ids, text_attention_mask, text_token_type_ids
    return text_input

def audio_wav2vec2(audio, wav2vec2, wav2vec2_processor):
    audio_len = 144000 #449
    # wav2vec2_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2_path)
    # wav2vec2 = Wav2Vec2Model.from_pretrained(args.wav2vec2_path)

    audio_fea = [0.0 for _ in range(audio_len)]
    if len(audio) >= audio_len:
        audio_fea[:] = audio[:audio_len]
    else:
        audio_fea[:len(audio)] = audio[:]
    assert len(audio_fea) == audio_len, 'Error with audio input length {} vs {}'.format(len(audio_fea), audio_len)

    audio_fea = wav2vec2_processor(audio_fea, sampling_rate=args.sample_rate, return_tensors='pt')
    audio_fea = wav2vec2(**audio_fea).last_hidden_state[0]
    # print(audio_fea.shape)
    '''
    audio_f = torch.zeros([audio_len, 768], dtype=torch.float)
    if len(audio_fea) > audio_len:
        audio_f[:, :] = audio_fea[:audio_len, :]
    else:
        audio_f[:len(audio_fea), :] = audio_fea[:, :]
    # print(audio_f)
    '''
    return audio_fea.detach().numpy() ## 转化成numpy.array

def audio_hubert(audio, hubert, wav2vec2_processor):
    audio_len = 144000 #449
    # wav2vec2_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2_path)
    # wav2vec2 = Wav2Vec2Model.from_pretrained(args.wav2vec2_path)

    audio_fea = [0.0 for _ in range(audio_len)]
    if len(audio) >= audio_len:
        audio_fea[:] = audio[:audio_len]
    else:
        audio_fea[:len(audio)] = audio[:]
    assert len(audio_fea) == audio_len, 'Error with audio input length {} vs {}'.format(len(audio_fea), audio_len)

    audio_fea = wav2vec2_processor(audio_fea, sampling_rate=args.sample_rate, return_tensors='pt')
    audio_fea = hubert(**audio_fea).last_hidden_state[0]
    # print(audio_fea.shape)

    return audio_fea.detach().numpy() ## 转化成numpy.array

def audio_mfcc(audio):
    ## 提取音频的特征
    audio_len = 144000 ##282
    # for (idx, example) in enumerate(data):
    audio_fea = [0.0 for _ in range(audio_len)]
    if len(audio) >= audio_len:
        audio_fea[:] = audio[:audio_len]
    else:
        audio_fea[:len(audio)] = audio[:]
    assert len(audio_fea) == audio_len, 'Error with audio input length {} vs {}'.format(len(audio), audio_len)
    audio_fea = np.array(audio_fea)
    mfcc = librosa.feature.mfcc(audio_fea, sr=16000, n_mfcc=13)  # mfcc系数
    mfcc = np.transpose(mfcc)
    # print(mfcc)

    """if len(mfcc) >= audio_len:
        audio_fea = mfcc[:audio_len]
    else:
        audio_fea = np.concatenate((mfcc,np.array([[0.0 for _ in range(13)] for _ in range(audio_len-len(mfcc))])), axis=0)"""
    return mfcc

def main():
    return 1


if __name__=="__main__":
    main()
