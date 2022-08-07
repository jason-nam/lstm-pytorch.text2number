import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

SOS_token = 0
EOS_token = 1

PATH = './data/rnn/'

class Tokenizer:
    def __init__(self):
        self.vocab2index = {"<SOS>": SOS_token, "<EOS>": EOS_token}
        self.index2vocab = {SOS_token: "<SOS>", EOS_token: "<EOS>"}
        self.n_vocab = len(self.vocab2index)

    def add_vocab(self, sentence):
        for word in sentence.split(" "):
            if word not in self.vocab2index:
                self.vocab2index[word] = self.n_vocab
                self.index2vocab[self.n_vocab] = word
                self.n_vocab += 1
    
    def to_seq(self, sentence):
        l = []
        for s in sentence.split(" "):
            l.append(self.vocab2index[s])
        return l

def model_load():
    """encoder"""
    encoder = torch.load(PATH + "encoder_model.pt")
    encoder.eval()

    """decoder"""
    decoder = torch.load(PATH + "decoder_model.pt")
    decoder.eval()

    """attention"""
    attention = torch.load(PATH + "attention_model.pt")
    attention.eval()

    return encoder, decoder, attention

def loader(df):
    src_tok = Tokenizer()
    for s in df['src'].values:
        for v in s.split(' '):
            src_tok.add_vocab(v)

    tar_tok = Tokenizer()
    for s in df['tar'].values:
        for v in s.split(' '):
            tar_tok.add_vocab(v)

    src_data = [src_tok.to_seq(s) for s in df['src'].values]
    tar_data = [[SOS_token] + tar_tok.to_seq(s) + [EOS_token] for s in df['tar'].values]

    # hparam
    hparam = {}
    hparam['embed_size'] = 8
    return src_tok, tar_tok, hparam, src_data, tar_data

class Encoder(nn.Module):
    def __init__(self, src_tok, hparam):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(src_tok.n_vocab, hparam['embed_size']) # embed_size = 4
        self.rnn = nn.LSTM(input_size=hparam['embed_size'], hidden_size=hparam['embed_size'])

    def forward(self, x, h, c):
        # (1)
        x = self.embed(x)
        # (embed_size)
        x = x.view((1, 1, -1))
        # (1,1,embed_size)
        x, (h, c) = self.rnn(x, (h, c))
        # (1,1,embed_size) (1,1,embed_size) (1,1,embed_size)
        return h, c

class Decoder(nn.Module):
    def __init__(self, tar_tok, hparam):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(tar_tok.n_vocab, hparam['embed_size']) # embed_size = 4
        self.rnn = nn.LSTM(input_size=hparam['embed_size'], hidden_size=hparam['embed_size'])

    def forward(self, x, h, c):
        # (1)
        x = self.embed(x)
        # (embed_size)
        x = x.view((1, 1, -1))
        # (1,1,embed_size)
        x, (h, c) = self.rnn(x, (h, c))
        # (1,1,embed_size) (1,1,embed_size) (1,1,embed_size)
        return h, c

class Attention(nn.Module):
    def __init__(self, tar_tok, hparam):
        super(Attention, self).__init__()
        self.wc = nn.Linear(hparam['embed_size'] * 2, hparam['embed_size']) # (embed_size * 2, embed_size) = (8, 4)
        self.tanh = nn.Tanh()
        self.wy = nn.Linear(hparam['embed_size'], tar_tok.n_vocab) # (embed_size, word_cnt)

    def forward(self, x):
        # (1,1,embed_size * 2)
        x = self.wc(x)
        # (1,1,embed_size)
        x = self.tanh(x)
        # (1,1,embed_size)
        x = self.wy(x)
        # (1,1,word_cnt)
        x = F.log_softmax(x, dim=2)
        # (1,1,word_cnt)
        return x

def train(src_tok, tar_tok, hparam, src_data, tar_data, df):
    encoder = Encoder(src_tok, hparam)
    decoder = Decoder(tar_tok, hparam)
    attention = Attention(tar_tok, hparam)

    enc_optimizer = optim.RMSprop(encoder.parameters(), lr=0.01)
    dec_optimizer = optim.RMSprop(decoder.parameters(), lr=0.01)
    att_optimizer = optim.RMSprop(attention.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    loss_hist = []
    for epoch in range(500):
        loss_avg = []

        for batch in range(len(src_data)):

            loss = 0

            src_train = torch.LongTensor(src_data[batch])

            h, c = torch.zeros((1, 1, hparam['embed_size'])), torch.zeros((1, 1, hparam['embed_size']))
            # (1,1,embed_size) (1,1,embed_size)

            enc_out = torch.Tensor([])
            # (src_len, 1, embed_size)
            for i in range(len(src_train)):
                # x = (1)
                h, c = encoder(src_train[i], h, c)
                # (1,1,embed_size) (1,1,embed_size)
                enc_out = torch.cat((enc_out, h))

            tar_train = torch.LongTensor(tar_data[batch])

            sent = []

            # teaching force rate
            rate = 0.5

            # teaching force
            if rate > np.random.rand():
                for i in range(len(tar_train[:-1])):
                    h, c = decoder(tar_train[i], h, c)
                    # (1,1,embed_size) (1,1,embed_size)
                    score = enc_out.matmul(h.view((1,hparam['embed_size'],1)))
                    # t 시점 state의 encoder h attention score
                    # (src_len, 1, 1) = score(hn, stT)
                    att_dis = F.softmax(score, dim=0)
                    # Attention Distribution
                    # (src_len,1,1)
                    att_v = torch.sum(enc_out * att_dis, dim=0).view(1,1,hparam['embed_size'])
                    # Attention Value
                    # (1,1,embed_size)
                    con = torch.cat((att_v, h), dim=2)
                    # Concatinate
                    out = attention(con)
                    # (1,1,word_cnt)
                    loss += criterion(out.view((1, -1)), tar_train[i+1].view(1))

                    sent.append(tar_tok.index2vocab[out.argmax().detach().item()])

            # without teaching force
            else:
                dec_in = tar_train[0]
                # skalar
                for i in range(len(tar_train[:-1])):
                    h, c = decoder(dec_in, h, c)
                    score = enc_out.matmul(h.view((1,hparam['embed_size'],1)))
                    att_dis = F.softmax(score, dim=0)
                    att_v = torch.sum(enc_out * att_dis, dim=0).view(1,1,hparam['embed_size'])
                    con = torch.cat((att_v, h), dim=2)
                    out = attention(con)
                    topv, topi = out.squeeze().topk(1) # detach!
                    # (1), (1)
                    dec_in = topi[0].detach()
                    # skalar
                    loss += criterion(out.view((1, -1)), tar_train[i+1].view(1))
                    
                    sent.append(tar_tok.index2vocab[out.argmax().detach().item()])
                    if dec_in == EOS_token:
                        break
            
            if (epoch + 1) % 50 == 0:
                print(epoch + 1, batch, loss.item())
                print(' '.join([tar_tok.index2vocab[t] for t in tar_train.detach().numpy()[1: ]]))
                print(' '.join(sent))
            
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            att_optimizer.zero_grad()

            loss = loss / len(df)
            loss.backward()

            enc_optimizer.step()
            dec_optimizer.step()
            att_optimizer.step()

            loss_avg.append(loss.item())

        loss_hist.append(sum(loss_avg))

        if (epoch + 1) % 50 == 0:
            print('avg loss', loss_hist[-1])
            print('=============================')

    torch.save(encoder, PATH + 'encoder_model.pt')
    torch.save(decoder, PATH + 'decoder_model.pt')
    torch.save(attention, PATH + 'attention_model.pt')

def evaluation(batch, src_tok, tar_tok, hparam, src_data, encoder, decoder, attention):

    src_train = torch.LongTensor(src_data[batch])

    h, c = torch.zeros((1, 1, hparam['embed_size'])), torch.zeros((1, 1, hparam['embed_size']))
    # (1,1,embed_size) (1,1,embed_size)

    enc_out = torch.Tensor([])
    # (src_len, 1, embed_size)
    for i in range(len(src_train)):
        # x = (1)
        h, c = encoder(src_train[i], h, c)
        # (1,1,embed_size) (1,1,embed_size)
        enc_out = torch.cat((enc_out, h))

    sent = []
    att_met = []

    # dec_in = tar_train[0]
    dec_in = torch.tensor(SOS_token)
    # print(dec_in)
    # print(len(tar_train[:-1]))

    # skalar
    for i in range(512):
        h, c = decoder(dec_in, h, c)
        score = enc_out.matmul(h.view((1,hparam['embed_size'],1)))
        att_dis = F.softmax(score, dim=0)

        att_met.append(att_dis.detach().numpy().reshape((-1)).tolist()) # visualization

        att_v = torch.sum(enc_out * att_dis, dim=0).view(1,1,hparam['embed_size'])
        con = torch.cat((att_v, h), dim=2)
        out = attention(con)
        topv, topi = out.squeeze().topk(1) # detach!
        # (1), (1)
        dec_in = topi[0].detach()
        # print(dec_in)
        sent.append(tar_tok.index2vocab[out.argmax().detach().item()])
        if dec_in == EOS_token:
            break
    
    s_arr = [src_tok.index2vocab[t] for t in src_train.detach().numpy()]
    
    print(' '.join(s_arr))
    # print(' '.join(t_arr))
    print(' '.join(sent))
    # print(tar_tok)

    return att_met, s_arr, sent
def main():

    df = pd.DataFrame(
    [
        ["사 일 구 혁명은 천 구백 육십 년 사 월 십 구 일에 학생과 시민이 중심 세력이 되어 일으킨 반독재 민주주의 운동이다.", "4.19혁명은 1960년 4월 19일에 학생과 시민이 중심 세력이 되어 일으킨 반독재 민주주의 운동이다."],
        ["코로나 일 구로 실내 활동이 늘어나면서, 스마트폰 이용자들의 모바일 앱 이용 시간이 전보다 네 시간가량 늘어났다.", "코로나19로 실내 활동이 늘어나면서, 스마트폰 이용자들의 모바일 앱 이용 시간이 전보다 4시간가량 늘어났다."],
        ["신신파스 아렉스는 국내 최초로 냉과 온, 두 번의 찜질 기능을 하나에 담아낸 파스라고 한다.", "신신파스 아렉스는 국내 최초로 냉과 온, 두 번의 찜질 기능을 하나에 담아낸 파스라고 한다."],
        ["아이들 방에는 미니 티슈를 주로 놓는데, 카카오 미니티슈가 품절이라 다른 브랜드로 두 개만 샀다.", "아이들 방에는 미니 티슈를 주로 놓는데, 카카오 미니티슈가 품절이라 다른 브랜드로 2개만 샀다."],
    ], 
    columns=['src', 'tar']
)
    sample_infer_data =['아이들 방에는 미니 티슈를 주로 놓는데, 카카오 미니티슈가 품절이라 다른 브랜드로 두 개만 샀다.',]
    src_tok, tar_tok, hparam, src_data, tar_data = loader(df=df)
    sample_data = [src_tok.to_seq(s) for s in sample_infer_data]
    print(sample_data)

    try:
        encoder, decoder, attention = model_load()
    except:
        train(src_tok=src_tok, tar_tok=tar_tok, src_data=src_data, tar_data=tar_data, hparam=hparam, df=df)
        encoder, decoder, attention = model_load()

    for i in range(len(sample_data)):
        _, s_arr, sent = evaluation(encoder=encoder, decoder=decoder, attention=attention, batch=i, src_data=sample_data, src_tok=src_tok, tar_tok=tar_tok, hparam=hparam)

        print("s_arr: ", s_arr)
        print("sent: ", sent)
        print()

if __name__ == "__main__":
    main()

