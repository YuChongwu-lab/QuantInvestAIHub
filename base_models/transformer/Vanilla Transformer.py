import math
import torch
import torch.nn as nn


#------------1.位置编码（Positional Encoding）----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        #pe:[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        #position: [max_len, 1]
        position = torch.arange(0, max_len,dtype= torch.float).unsqueeze(1)# [max_len, 1]
        # div_term: [d_model/2]
        div_term = torch.exp(
            torch.arange(0,d_model, 2).float()*(-math.log(10000.0)/d_model))
        # pe[:, 0::2]: [max_len, d_model/2] 偶数维
        pe[:,0::2] = torch.sin(position * div_term)
        # pe[:, 1::2]: [max_len, d_model/2] 奇数维
        pe[:,1::2] = torch.cos(position * div_term)

        # 加 batch 维度: [1, max_len, d_model],方便与x: [batch, seq_len, d_model] 广播相加
        pe = pe.unsqueeze(0)
        # 注册为 buffer，不参与训练，但会随模型保存/搬运
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #x:[batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # self.pe[:, :seq_len] : [1, seq_len, d_model]，广播到 batch 维
        x = x + self.pe[:, :seq_len]

        # 输出仍然是 [batch, seq_len, d_model]
        return self.dropout(x)


#---------------2.多头注意力（Multi-Head Attention）---------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads ==0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads# 每个 head 的维度

        #W_q/W_k/W_v: 线性层，输入 [*, d_model] -> 输出 [*, d_model]
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性层:[*, d_model] -> [*, d_model]
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask=None):
        # query/key/value: [batch, seq_len, d_model]
        batch_size = query.size(0)

        # 线性映射 + reshape 成多头格式,并调整维度[batch, heads, seq_len, d_k]
        # 线性映射后 shape 仍然是 [batch, seq_len, d_model]
        # 再 view -> [batch, seq_len, num_heads, d_k]
        # 再transpose：-> [batch, num_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力得分：Q·K^T / sqrt(d_k)
        #scores: [batch, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_k)

        # mask 处理（例如 decoder 遮未来 token）
        #若存在，应能广播到 [batch, num_heads, seq_len_q, seq_len_k]
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        # softmax 得到注意力权重
        #attn: [batch, num_heads, seq_len_q, seq_len_k]
        attn = torch.softmax(scores, dim=-1)

        # 加权求值
        # context: [batch, num_heads, seq_len_q, d_k]
        context = torch.matmul(attn,V)

        #合并heads
        # context.transpose: [batch, seq_len_q, num_heads, d_k]
        # contiguous().view: [batch, seq_len_q, num_heads * d_k] = [batch, seq_len_q, d_model]
        context = context.transpose(1,2).contiguous().view(batch_size, -1,self.num_heads * self.d_k)


        #输出线性层
        #输出: [batch, seq_len_q, d_model]
        out= self.W_o(context)

        return self.dropout(out)


#--------------------3.前馈网络（Feed Forward Network）-----------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # fc1: [batch, seq_len, d_model] -> [batch, seq_len, d_ff]
        self.fc1 = nn.Linear(d_model, d_ff)
        # fc2: [batch, seq_len, d_ff] -> [batch, seq_len, d_model]
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)#FFN中间dropout

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # fc1(x): [batch, seq_len, d_ff]
        # fc2(...): [batch, seq_len, d_model]
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


#-----------4.编码器层（ Encoder Layer）----------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads,d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: [batch, src_len, d_model]

        # Self-Attention + 残差连接
        # self_attn 输出: [batch, src_len, d_model]
        attn_output,_ = self.attn(x, x, x, mask)
        # x + attn_output: [batch, src_len, d_model]
        x = self.norm1(x +attn_output)

        #FNN + 残差连接
        # ffn_output: [batch, src_len, d_model]
        ffn_output = self.ffn(x)
        # x + ffn_output: [batch, src_len, d_model]
        x = self.norm2(x + ffn_output)

        # 返回: [batch, src_len, d_model]
        return x


#---------5.解码器层(Decoder Layer)---------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask =None, memory_mask=None):
        # x: [batch, tgt_len, d_model]
        # enc_output: [batch, src_len, d_model]

        # Masked Self-Attention（遮住未来 token）
        # self_attn(x,x,x): 输出 [batch, tgt_len, d_model]
        attn1 = self.self_attn(x, x, x, tgt_mask)
        # x + attn1: [batch, tgt_len, d_model]
        x = self.norm1(x +attn1)

        # Encoder-Decoder Cross Attention（使用 Encoder 输出）
        # cross_attn: Q 来自 x([batch, tgt_len, d_model])
        #             K,V 来自 enc_output([batch, src_len, d_model])
        # 输出: [batch, tgt_len, d_model]
        attn2 = self.cross_attn(x, enc_output, enc_output, memory_mask)
        # x + attn2: [batch, tgt_len, d_model]
        x = self.norm2(x + attn2)

        #FFN
        # ffn_output: [batch, tgt_len, d_model]
        ffn_output = self.ffn(x)
        # x + ffn_output: [batch, tgt_len, d_model]
        x = self.norm3(x + ffn_output)

        # 返回: [batch, tgt_len, d_model]
        return x

#---------6.编码器(Encoder)------------------------
class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,num_layers, num_heads, d_ff):
        super().__init__()
        # embedding: [batch, seq_len] -> [batch, seq_len, d_model]
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos = PositionalEncoding(d_model)

        # layers: 长度为 num_layers 的 EncoderLayer 列表
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # x: [batch, src_len]
        # embedding(x): [batch, src_len, d_model]
        x = self.embedding(x)
        # 位置编码后仍为 [batch, src_len, d_model]
        x = self.pos(x)

        # 逐层编码，每层输入输出都是 [batch, src_len, d_model]
        for layer in self.layers:
            x = layer(x, mask)

        # 返回编码结果: [batch, src_len, d_model]
        return x

#------------8.解码器(Decoder)--------
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
        super().__init__()
        # embedding: [batch, tgt_len] -> [batch, tgt_len, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # fc_out: [batch, tgt_len, d_model] -> [batch, tgt_len, vocab_size]
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # x: [batch, tgt_len]
        # enc_output: [batch, src_len, d_model]

        # embedding(x): [batch, tgt_len, d_model]
        x = self.embedding(x)
        # 加位置编码: [batch, tgt_len, d_model]
        x = self.pos(x)

        # 逐层解码，每层输入输出: [batch, tgt_len, d_model]
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)

        # 输出 [batch, tgt_len, vocab_size]
        return self.fc_out(x)

#-----------------8.完整Transformer --------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048):
        super().__init__()

        # encoder 输出: [batch, src_len, d_model]
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads,d_ff)
        # decoder 输出: [batch, tgt_len, vocab_size]
        self.decoder = Decoder(vocab_size, d_model, num_layers, num_heads,d_ff)

    def make_subsequent_mask(self, size):
        """Decoder 用的下三角 mask，避免看到未来 token"""
        # torch.ones(size, size): [size, size]
        # torch.tril(...): [size, size] 下三角(含对角线)为1，上三角为0
        # unsqueeze(0).unsqueeze(0): [1, 1, size, size]，可广播到 [batch, heads, size, size]
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
        return mask# [1, 1, size, size]

    def forward(self, src, tgt):
        # src: [batch, src_len]
        # tgt: [batch, tgt_len]

        # enc_output: [batch, src_len, d_model]
        enc_output = self.encoder(src)

        # tgt_mask: [1, 1, tgt_len, tgt_len]，在注意力中广播到 [batch, heads, tgt_len, tgt_len]
        tgt_mask = self.make_subsequent_mask(tgt.size(1))

        # decoder 输出: [batch, tgt_len, vocab_size]
        out = self.decoder(tgt, enc_output, tgt_mask = tgt_mask)

        return out

#-----------------测试运行---------------------
if __name__ == "__main__":
    vocab_size = 10000

    model = Transformer(vocab_size)

    # src: [batch=2, src_len=10]
    src = torch.randint(0,vocab_size, (2, 10))
    # tgt: [batch=2, tgt_len=10]
    tgt = torch.randint(0, vocab_size, (2, 10))

    # out: [2, 10, vocab_size]
    out = model(src, tgt)

    print("out.shape:", out.shape) # # 应输出: torch.Size([2, 10, 10000])

















