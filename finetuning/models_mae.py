import torch
import torch.nn as nn
import numpy as np

from vision_transformer import OriginVisionTransformer
import torch.nn.functional as F
from torch.autograd import Variable


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, compression):
        super().__init__()
        self.compression = compression
        if self.compression != 3:
            self.encoder = Encoder(1024)
        else:
            self.encoder = Encoder(512)

        self.decoder = HybirdDecoder(vocab_size=vocab_size)
        self.prediction = nn.Linear(128, vocab_size)
        self.max_len = max_len

        if self.compression != 3:
            self.pos_embed = nn.Parameter(torch.zeros(1, 20, 1024))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, 20, 512))

    def forward_train(self, x, y):
        x = x.view(x.shape[0], 7, 20, 512)
        if self.compression == 1:
            x1 = torch.mean(x[:, :3, :, :], dim=1, keepdim=False)
            x2 = torch.mean(x[:, 3:, :, :], dim=1, keepdim=False)
            x = torch.cat([x1, x2], dim=-1)
        elif self.compression == 2:
            x1 = torch.mean(x[:, ::2, :, :], dim=1, keepdim=False)
            x2 = torch.mean(x[:, 1::2, :, :], dim=1, keepdim=False)
            x = torch.cat([x1, x2], dim=-1)
        elif self.compression == 3:
            x = torch.mean(x, dim=1, keepdim=False)

        x = x + self.pos_embed
        encoder_outputs = self.encoder(x)

        vocab_out = self.decoder.forward_train(encoder_outputs, self.max_len, y)
        vocab_out = self.prediction(vocab_out)
        return vocab_out

    def forward_test(self, x):
        x = x.view(x.shape[0], 7, 20, 512)
        if self.compression == 1:
            x1 = torch.mean(x[:, :3, :, :], dim=1, keepdim=False)
            x2 = torch.mean(x[:, 3:, :, :], dim=1, keepdim=False)
            x = torch.cat([x1, x2], dim=-1)
        elif self.compression == 2:
            x1 = torch.mean(x[:, ::2, :, :], dim=1, keepdim=False)
            x2 = torch.mean(x[:, 1::2, :, :], dim=1, keepdim=False)
            x = torch.cat([x1, x2], dim=-1)
        elif self.compression == 3:
            x = torch.mean(x, dim=1, keepdim=False)

        x = x + self.pos_embed
        encoder_outputs = self.encoder(x)

        outputs = []
        batch_size = x.size(0)
        input = torch.zeros([batch_size]).long()
        input = input.cuda()

        last_hidden = Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size))
        last_hidden = last_hidden.cuda()

        for i in range(self.max_len - 1):
            output, last_hidden = self.decoder.forward_step(input, last_hidden, encoder_outputs)
            output = self.prediction(output)
            input = output.max(1)[1]
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class Encoder(nn.Module):
    """
    input: [batch_size, 32, 256]
    output: [batch_size, 32, 128]
    """

    def __init__(self, embed_size, num_rnn_layers=2, rnn_hidden_size=128, dropout=0.5):
        super(Encoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.gru = nn.GRU(embed_size, rnn_hidden_size, num_rnn_layers,
                          batch_first=True,
                          dropout=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size))
        h0 = h0.cuda()
        out, hidden = self.gru(x, h0)

        return out


class HybirdDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_rnn_layers=2, dropout=0.5):
        super(HybirdDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_rnn_layers = num_rnn_layers

        self.attn = DotProductAttentionLayer()
        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_rnn_layers, batch_first=True,
                          dropout=dropout)

        self.wc = nn.Linear(2 * hidden_size, hidden_size)

        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward_train(self, encoder_outputs, max_len, y):
        batch_size = encoder_outputs.size(0)
        last_hidden = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size))
        last_hidden = last_hidden.cuda()

        input = y[:, :max_len - 1]  # [batch, max_len-1]
        embed_input = self.embedding(input)  # [batch, max_len-1, 128]
        query, _ = self.gru(embed_input, last_hidden)  # [batch, max_len-1, 128]
        key = encoder_outputs  # [batch, 32, 128]
        value = encoder_outputs  # [batch, 32, 128]

        weighted_context = self.attn(query, key, value)  # [batch, max_len-1, 128]
        output = self.tanh(self.wc(torch.cat((query, weighted_context), 2)))  # [batch, max_len-1, 128]

        return output

    def forward_step(self, input, last_hidden, encoder_outputs):
        embed_input = self.embedding(input)
        output, hidden = self.gru(embed_input.unsqueeze(1), last_hidden)
        output = output.squeeze(1)

        query = output.unsqueeze(1)  # [batch, 1, 128]
        key = encoder_outputs  # [batch, 32, 128]
        value = encoder_outputs  # [batch, 32, 128]

        weighted_context = self.attn(query, key, value).squeeze(1)
        output = self.tanh(self.wc(torch.cat((output, weighted_context), 1)))
        return output, hidden


class DotProductAttentionLayer(nn.Module):
    def __init__(self):
        super(DotProductAttentionLayer, self).__init__()

    def forward(self, query, key, value):
        logits = torch.matmul(query, key.permute(0, 2, 1))  # [len, 256]*[256, 32]=[len, 32]

        alpha = F.softmax(logits, dim=-1)
        weighted_context = torch.matmul(alpha, value)  # [len, 32] * [32, 256]=[len, 256]

        return weighted_context


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio, qkv_bias, norm_layer):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, num_heads, qkv_bias, norm_layer)
        self.dec_enc_attn = MultiHeadAttention(d_model, num_heads, qkv_bias, norm_layer)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, mlp_ratio, norm_layer)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, qkv_bias, norm_layer):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_model * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_model * num_heads, bias=qkv_bias)
        self.fc = nn.Linear(num_heads * d_model, d_model, bias=qkv_bias)
        self.norm = norm_layer(d_model)
        self.attention = ScaledDotProductAttention(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.num_heads, self.d_model).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.num_heads, self.d_model).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.num_heads, self.d_model).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context = self.attention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_model)
        output = self.fc(context)
        return self.norm(output + residual)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, mlp_ratio, norm_layer):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * mlp_ratio, d_model, bias=False)
        )
        self.norm = norm_layer(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.norm(output + residual)


class VisionTransformer(OriginVisionTransformer):
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

    def forward(self, x):
        encoder_out = self.forward_features(x)
        return encoder_out

    def forward_features(self, x):
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        outcome = self.norm(x)

        return outcome
