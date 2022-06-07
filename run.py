import os
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from args_setting import args
from data_process import data_processing
from transformers import BertTokenizer, Wav2Vec2Processor, Wav2Vec2Model, HubertModel
from model_bert_wav2vec import audio_text_model
from get_audio_text_input import text_bert, audio_mfcc, audio_wav2vec2, audio_hubert
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
import librosa
import numpy as np
import pickle as pkl
import logging
import random

## 创建日志输出到文件和操作台
def creat_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formater = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ## 创建handler将日志输出到文档
    file_handler = logging.FileHandler(filename=args.log_dir+'/train.log')
    file_handler.setFormatter(formater)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    ## 创建handler将日志输出到操作台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formater)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    return logger

class struct_inputs(object):
    def __init__(self, audio_inputs, text_ids, text_attention_mask, text_token_type_ids, labels):
        self.audio_inputs = audio_inputs
        self.text_ids = text_ids
        self.text_attention_mask = text_attention_mask
        self.text_token_type_ids = text_token_type_ids
        self.labels = labels

def get_dataset(data, logger, data_type ,text_type, audio_type):
    logger.info('Start creat {} {}_{}_dataset '.format(data_type, text_type, audio_type))
    # inputs_path = os.path.join(
    #     args.processed_data_path + '/{}_inputs_dataset2'.format(data_type))
    inputs_path = os.path.join(args.processed_data_path + '/{}_{}_{}_dataset'.format(data_type, text_type, audio_type))
    labels = []
    if not os.path.exists(inputs_path):
        text_ids = []
        text_attention_mask = []
        text_token_type_ids = []
        audio_inputs = []
        wav2vec2_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2_path)
        hubert = HubertModel.from_pretrained('/workspace/practice/双模（语音文本）/IEMOCAP_Emotion/pretrained_model/hubert-base-ls960')
        wav2vec2 = Wav2Vec2Model.from_pretrained(args.wav2vec2_path)
        for (idx, example) in enumerate(data):
            labels.append(example.label)
            if text_type == 'bert':
                # text_ids, text_attention_mask, text_token_type_ids = text_bert(data)
                text_input = text_bert(example.text)
                text_ids.append(text_input['input_ids'])
                text_attention_mask.append(text_input['attention_mask'])
                text_token_type_ids.append(text_input['token_type_ids'])
            if audio_type == 'wav2vec2':
                print(idx)
                audio_inputs.append(audio_wav2vec2(example.audio, wav2vec2, wav2vec2_processor))
            elif audio_type == 'mfcc':
                audio_inputs.append(audio_mfcc(example.audio))
            elif audio_type == 'hubert':
                print(idx)
                audio_inputs.append(audio_hubert(example.audio, hubert, wav2vec2_processor))

        text_ids = torch.tensor(text_ids, dtype=torch.long)
        text_attention_mask = torch.tensor(text_attention_mask, dtype=torch.long)
        text_token_type_ids = torch.tensor(text_token_type_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        audio_inputs = torch.tensor(audio_inputs, dtype=torch.float)
        print(audio_inputs.shape)
        inputs = struct_inputs(audio_inputs, text_ids, text_attention_mask, text_token_type_ids, labels)
        torch.save(inputs, inputs_path)
    f = torch.load(inputs_path)

    #print(f.audio_inputs)
    dataset = TensorDataset(f.audio_inputs, f.text_ids, f.text_attention_mask, f.text_token_type_ids, f.labels)
    logger.info('Finish creat {} {}_{}_dataset '.format(data_type, text_type, audio_type))
    return dataset


'''
## 获取bert的文本输入、获wav2vec2编码的音频特征
def get_dataset1(data, logger, data_type):
    logger.info('Start creat {} dataset type 1'.format(data_type))
    inputs_path = os.path.join(args.processed_data_path + '/{}_inputs_dataset1_1'.format(data_type, args.text_max_len, args.audio_max_len))

    if not os.path.exists(inputs_path):
        text_ids = []
        text_attention_mask = []
        text_token_type_ids = []
        audio_inputs = []
        labels = []
        tokenizer = BertTokenizer.from_pretrained(args.bert_path) ## 创建bert的输入
        ## 提取音频的特征
        wav2vec2_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2_path)
        wav2vec2 = Wav2Vec2Model.from_pretrained(args.wav2vec2_path)
        # wav2vec2.to(args.device)
        for (idx, example) in enumerate(data):
            text_input = tokenizer.encode_plus(
                example.text,
                add_special_tokens=True,
                max_length=args.text_max_len,
                padding='max_length',
                truncation=True  ## 保证超过max_length的句子自动截断
            )
            print(idx)
            assert len(text_input['input_ids']) == args.text_max_len, "Error with text input length {} vs {}".format(len(text_input['input_ids']), args.text_max_len)
            assert len(text_input['attention_mask']) == args.text_max_len, "Error with text input length {} vs {}".format(len(text_input['attention_mask']), args.text_max_len)
            assert len(text_input['token_type_ids']) == args.text_max_len, "Error with text input length {} vs {}".format(len(text_input['token_type_ids']), args.text_max_len)
            text_ids.append(text_input['input_ids'])
            text_attention_mask.append(text_input['attention_mask'])
            text_token_type_ids.append(text_input['token_type_ids'])
            
            # audio_fea = [0.0]*args.audio_max_len
            # if len(example.audio) >= args.audio_max_len:
            #     audio_fea[:] = example.audio[:args.audio_max_len]
            # else:
            #     audio_fea[:len(example.audio)] = example.audio[:]
            # assert len(audio_fea) == args.audio_max_len, 'Error with audio input length {} vs {}'.format(len(audio_fea), args.audio_max_len)
            
            audio_fea = wav2vec2_processor(example.audio, sampling_rate=args.sample_rate, return_tensors='pt')
            audio_fea = wav2vec2(**audio_fea).last_hidden_state[0]
            # print(audio_fea[0])
            audio_f = torch.zeros([args.audio_max_len, 768], dtype=torch.float)
            if len(audio_fea) > args.audio_max_len:
                audio_f[:,:] = audio_fea[:args.audio_max_len, :]
            else:
                audio_f[:len(audio_fea), :] = audio_fea[:, :]
            audio_fea = audio_f
            audio_inputs.append(audio_fea.detach().numpy())

            #print(audio_fea)
            #print(torch.tensor(audio_fea,dtype=torch.float))
            labels.append(example.label)
            #all_inputs.append(struct_inputs(audio_fea, text_input['input_ids'], text_input['attention_mask'], text_input['token_type_ids'], example.label))
            #print(idx)
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        text_attention_mask = torch.tensor(text_attention_mask, dtype=torch.long)
        text_token_type_ids = torch.tensor(text_token_type_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        audio_inputs = torch.tensor(audio_inputs, dtype=torch.float)
        print(audio_inputs.shape)
        inputs = struct_inputs(audio_inputs, text_ids, text_attention_mask, text_token_type_ids, labels)
        torch.save(inputs, inputs_path)

    f = torch.load(inputs_path)
    #print(f.audio_inputs)
    dataset = TensorDataset(f.audio_inputs, f.text_ids, f.text_attention_mask, f.text_token_type_ids, f.labels)
    logger.info('Finish creat {} dataset'.format(data_type))
    return dataset

## 获取bert的文本输入、获mfcc特征
def get_dataset2(data, logger, data_type):
    logger.info('Start creat {} dataset type 2'.format(data_type))
    inputs_path = os.path.join(args.processed_data_path + '/{}_inputs_dataset2'.format(data_type, args.text_max_len, args.audio_max_len))

    if not os.path.exists(inputs_path):
        text_ids = []
        text_attention_mask = []
        text_token_type_ids = []
        audio_inputs = []
        labels = []
        tokenizer = BertTokenizer.from_pretrained(args.bert_path) ## 创建bert的输入
        for (idx, example) in enumerate(data):
            text_input = tokenizer.encode_plus(
                example.text,
                add_special_tokens=True,
                max_length=args.text_max_len,
                padding='max_length',
                truncation=True  ## 保证超过max_length的句子自动截断
            )
            print(idx)
            assert len(text_input['input_ids']) == args.text_max_len, "Error with text input length {} vs {}".format(len(text_input['input_ids']), args.text_max_len)
            assert len(text_input['attention_mask']) == args.text_max_len, "Error with text input length {} vs {}".format(len(text_input['attention_mask']), args.text_max_len)
            assert len(text_input['token_type_ids']) == args.text_max_len, "Error with text input length {} vs {}".format(len(text_input['token_type_ids']), args.text_max_len)
            text_ids.append(text_input['input_ids'])
            text_attention_mask.append(text_input['attention_mask'])
            text_token_type_ids.append(text_input['token_type_ids'])

            ## 提取音频的特征
            audio = [0.0]*args.audio_max_len
            if len(example.audio) >= args.audio_max_len:
                audio[:] = example.audio[:args.audio_max_len]
            else:
                audio[:len(example.audio)] = example.audio[:]
            assert len(audio) == args.audio_max_len, 'Error with audio input length {} vs {}'.format(len(audio_fea), args.audio_max_len)
            audio = np.array(audio)
            mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=13) # mfcc系数

            audio_fea = np.transpose(mfcc)  # 因为输出的数据是（num_mfcc,t),所以需要进行一个转置
            # print(audio_fea.shape)

            audio_inputs.append(audio_fea)

            #print(audio_fea)
            #print(torch.tensor(audio_fea,dtype=torch.float))
            labels.append(example.label)
            #all_inputs.append(struct_inputs(audio_fea, text_input['input_ids'], text_input['attention_mask'], text_input['token_type_ids'], example.label))
            #print(idx)
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        text_attention_mask = torch.tensor(text_attention_mask, dtype=torch.long)
        text_token_type_ids = torch.tensor(text_token_type_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        audio_inputs = torch.tensor(audio_inputs, dtype=torch.float)
        print(audio_inputs.shape)
        inputs = struct_inputs(audio_inputs, text_ids, text_attention_mask, text_token_type_ids, labels)
        torch.save(inputs, inputs_path)

    f = torch.load(inputs_path)
    #print(f.audio_inputs)
    dataset = TensorDataset(f.audio_inputs, f.text_ids, f.text_attention_mask, f.text_token_type_ids, f.labels)
    logger.info('Finish creat {} dataset'.format(data_type))
    return dataset
'''

def train(train_dataloader, dev_dataloader, test_dataloader, logger):
    logger.info('--------Start train--------')
    device = args.device
    model = audio_text_model()
    optimizer = Adam(model.parameters(), lr=0.0001)
    model.to(device)
    best_acc = 0
    #print(model)
    # for para in model.parameters():
    #     print(para)
    for epoch in range(args.epoch):
        model.train()
        correct = 0
        total = 0
        e_loss = 0
        train_iter = tqdm(train_dataloader)
        #loss = torch.zeros(1).to(device)
        for (idx, batch) in enumerate(train_iter):
            optimizer.zero_grad()
            audio_inputs = batch[0].to(device)
            # print(audio_inputs)
            text_ids = batch[1].to(device)
            text_attention_mask = batch[2].to(device)
            text_token_type_ids = batch[3].to(device)
            labels = batch[4].to(device)

            output = model(audio_inputs, text_ids, text_attention_mask, text_token_type_ids)

            loss = F.cross_entropy(output, labels)
            e_loss += loss.item()
            loss.backward()
            optimizer.step()
            '''
            if (idx+1)%8 == 0 or (idx+1) == len(train_dataloader):
                loss.backward()
                optimizer.zero_grad()
                loss = torch.zeros(1).to(device)
            '''

            predict = torch.argmax(output, dim=1)
            correct += (predict == labels).sum().item()
            total += len(labels)
            #print(correct, total)
            train_acc = correct/total
            train_iter.set_description('Epoch:{} batch:{} acc:{:.4f}% loss:{:.4f} '.format(epoch+1, idx+1, train_acc*100, loss.item()))
        train_loss = e_loss/len(train_dataloader)
        dev_acc, dev_loss = evaluate(dev_dataloader, model)
        logger.info('Epoch:{} train_acc:{}% train_loss:{} dev_acc:{} dev_loss:{}'.format(epoch+1, train_acc*100, train_loss, dev_acc*100, dev_loss))
        if dev_acc > best_acc:
            best_acc = dev_acc
        test(test_dataloader, model, logger)
        # else:
        #     logger.info('Break train for dev_acc have down')
        #     saved_path = args.saved_model_path + 'epoch_{}_model.pth'.format(epoch)
        #     torch.save(model, saved_path)
        #     logger.info('Model have been saved')
    logger.info('--------Finish train--------')
    test(test_dataloader, model, logger)

def evaluate(dev_dataloader, model):
    model.eval()
    device = args.device
    loss = 0
    correct = 0
    total = 0
    dev_iter = tqdm(dev_dataloader)
    for (idx, batch) in enumerate(dev_iter):
        audio_inputs = batch[0].to(device)
        text_ids = batch[1].to(device)
        text_attention_mask = batch[2].to(device)
        text_token_type_ids = batch[3].to(device)
        labels = batch[4].to(device)

        output = model(audio_inputs, text_ids, text_attention_mask, text_token_type_ids)

        loss += F.cross_entropy(output, labels).item()
        total += len(labels)
        predict = torch.argmax(output, dim=1)
        correct += (predict == labels).sum().item()
        dev_iter.set_description('dev batch:{} acc:{:.4f}% loss:{:.4f} '.format(idx+1, correct/total *100, loss/(idx+1)))
    return correct/total, loss/len(dev_dataloader)

def test(test_dataloader, model, logger):
    model.eval()
    device = args.device
    predict = []
    true = []
    correct = 0
    test_iter = tqdm(test_dataloader)
    for (idx, batch) in enumerate(test_iter):
        audio_inputs = batch[0].to(device)
        text_ids = batch[1].to(device)
        text_attention_mask = batch[2].to(device)
        text_token_type_ids = batch[3].to(device)
        labels = batch[4].to(device)

        output = model(audio_inputs, text_ids, text_attention_mask, text_token_type_ids)
        pred = torch.argmax(output, dim=1)
        correct += (pred == labels).sum().item()

        predict.extend(pred.cpu().numpy())
        true.extend(batch[4].numpy())

    print('*** test acc :{:.4f}%'.format(correct/len(true)*100))
    logger.info('test classification_report: {}'.format(classification_report(true, predict)))



if __name__=="__main__":
    logger = creat_logger(args)
    args.dir_now = os.getcwd()
    torch.manual_seed(2)
    ## 如果没有预处理文件，进行预处理
    if not os.path.exists(os.path.join(args.processed_data_path + '/' + 'process_data')):
        logger.info('Start processing data')
        processed_data = data_processing('./data', logger)
        torch.save(processed_data, os.path.join(args.processed_data_path + '/' + 'process_data'))
        logger.info('Finish processing data')
    processed_data = torch.load(args.processed_data_path + '/' + 'process_data')
    random.shuffle(processed_data)

    if torch.cuda.is_available():
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    train_data = processed_data[ :int(len(processed_data)*0.7)]
    dev_data = processed_data[int(len(processed_data)*0.7): int(len(processed_data)*0.8)]
    test_data = processed_data[int(len(processed_data)*0.8): ]
    logger.info('Nums of train_dataset : dev_dataset : test_dataset = {} : {} : {}'.format(len(train_data), len(dev_data), len(test_data)))

    dev_dataset = get_dataset(dev_data, logger, 'dev', text_type=args.text_type, audio_type=args.audio_type)
    train_dataset = get_dataset(train_data, logger, 'train', text_type=args.text_type, audio_type=args.audio_type)
    test_dataset = get_dataset(test_data, logger, 'test', text_type=args.text_type, audio_type=args.audio_type)

    # dev_dataset = get_dataset2(dev_data, logger, 'dev')
    # train_dataset = get_dataset2(train_data, logger, 'train')
    # test_dataset = get_dataset2(test_data, logger, 'test')


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    train(train_dataloader, dev_dataloader, test_dataloader, logger)
