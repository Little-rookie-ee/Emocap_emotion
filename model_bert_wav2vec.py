import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import math
from transformers import BertModel, Wav2Vec2Model, Wav2Vec2Processor
from args_setting import args


class audio_text_model(nn.Module):
    def __init__(self):
        super(audio_text_model, self).__init__()

        self.bertmodel = BertModel.from_pretrained(args.bert_path)
        #self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2_path)
        # self.wav2vec2 = Wav2Vec2Model.from_pretrained(args.wav2vec2_path)
        self.lstm_text = nn.LSTM(input_size=768, hidden_size=128, batch_first=True)
        self.lstm_audio = nn.LSTM(input_size=768, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(256, 128)

        self.fc2 = nn.Linear(128, 6)

        self.q = nn.Linear(128, 128)
        self.k = nn.Linear(128, 128)
        self.v = nn.Linear(128, 128)

        ##  初始化attention权重， 计算方式：https://blog.csdn.net/qsmx666/article/details/107118550
        ## text attention
        self.w_omega1 = nn.Parameter(torch.Tensor(
            64 * 2, 64 * 2))
        self.u_omega1 = nn.Parameter(torch.Tensor(64 * 2, 1))

        nn.init.uniform_(self.w_omega1, -0.1, 0.1)
        nn.init.uniform_(self.u_omega1, -0.1, 0.1)

        ## audio attention
        self.w_omega2 = nn.Parameter(torch.Tensor(
            64 * 2, 64 * 2))
        self.u_omega2 = nn.Parameter(torch.Tensor(64 * 2, 1))

        nn.init.uniform_(self.w_omega2, -0.1, 0.1)
        nn.init.uniform_(self.u_omega2, -0.1, 0.1)

    def text_bilstm_att(self, text_ids, text_attention_mask, text_token_type_ids):
        text_output = self.bertmodel(input_ids=text_ids, attention_mask = text_attention_mask,token_type_ids=text_token_type_ids)
        text_output = text_output.last_hidden_state  ## size (batch, sequence_length, hidden_dim) hiddem_dim = 768

        text_out, _ = self.lstm_text(text_output) ## size (batch, sequence_length, lstm_hidden_dim)

        x = text_out
        ## Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega1)) ## size (batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega1) ## size (batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)  ## size (batch_size, seq_len, 1)
        scored_x = x * att_score ## size (batch_size, seq_len, 2 * num_hiddens)

        outs = torch.sum(scored_x, dim=1)  # 加权求和, size (batch_size, 2 * num_hiddens)
        # outs = self.fc3(feat)
        # outs = self.fc2(outs)  ## (batch_size, 4)
        return outs
    def audio_bilstm_att(self, audio_inputs):
        audio_out, _ = self.lstm_audio(audio_inputs)

        x = audio_out
        ## Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega2)) ## size (batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega2) ## size (batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)  ## size (batch_size, seq_len, 1)
        scored_x = x * att_score ## size (batch_size, seq_len, 2 * num_hiddens)

        outs = torch.sum(scored_x, dim=1)  # 加权求和, size (batch_size, 2 * num_hiddens)
        # outs = self.fc4(feat)
        # outs = self.fc2(outs)  ## (batch_size, 4)
        return outs

    ## 使用attention融合语音于文本的信息，Q来自audio， K，V来自text
    def text_audio_att(self, audio_inputs, text_ids, text_attention_mask, text_token_type_ids):
        text_output = self.bertmodel(input_ids=text_ids, attention_mask = text_attention_mask,token_type_ids=text_token_type_ids)
        text_output = text_output.last_hidden_state  ## size (batch, sequence_length, hidden_dim) hiddem_dim = 768

        text_out, _ = self.lstm_text(text_output)
        audio_out, _ = self.lstm_audio(audio_inputs)

        Q = self.q(audio_out)
        K = self.k(text_out)
        V = self.v(text_out)

        attention_scores = torch.matmul(Q, K.transpose(2, 1))
        attention_scores = attention_scores / math.sqrt(128)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        out = torch.matmul(attention_probs, V)
        # print(out.shape)
        out = torch.sum(out, dim=1)
        # print(out)
        out = self.fc2(out)
        return out

    def forward(self, audio_inputs, text_ids, text_attention_mask, text_token_type_ids):
        """text_output = self.bertmodel(input_ids=text_ids, attention_mask = text_attention_mask,token_type_ids=text_token_type_ids)
        text_output = text_output.last_hidden_state  ## size (batch, sequence_length, hidden_dim) hiddem_dim = 768

        text_out, _ = self.lstm_text(text_output) ## size (batch, sequence_length, lstm_hidden_dim)
        text_out = text_out[:, -1, :]  ## size (batch, 1, lstm_hidden_dim)
        text_out = torch.squeeze(text_out,dim=1)  ## size (batch, lstm_hidden_dim)

        # print(audio_inputs.size)
        # audio_inputs = self.norm(audio_inputs)
        # print(audio_inputs[:30])
        # print(audio_inputs)
        audio_out, _ = self.lstm_audio(audio_inputs)  ## size (batch, sequence_length, lstm_audio_hidden_dim)
        audio_out = audio_out[:, -1, :]  ## size (batch, 1, lstm_hidden_dim)
        audio_out = torch.squeeze(audio_out,dim=1)  ## size (batch, lstm_hidden_dim)
        # assert text_out.shape == audio_out.shape, 'Error output shape of text and audio {} vs {}'.format(text_out.shape, audio_out.shape)

        out = torch.cat((audio_out, text_out), 1) ## size (batch, 2*hidden_dim)
        out = self.fc1(out)
        # print(audio_out)
        out = self.fc2(out)
        # print(out)
        out = F.softmax(out, dim=1)"""

        # text_out = self.text_bilstm_att(text_ids, text_attention_mask, text_token_type_ids)
        # audio_out = self.audio_bilstm_att(audio_inputs)
        # out = torch.cat((audio_out, text_out), 1)  ## size (batch, 2*hidden_dim)
        # out = self.fc1(text_out)
        # out = self.fc2(audio_out)
        # out = F.softmax(out, dim=1)

        out = self.text_audio_att(audio_inputs, text_ids, text_attention_mask, text_token_type_ids)
        return out

