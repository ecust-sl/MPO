from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import pack_wrapper, AttModel
# from .atten_prefer import EnhancedDynamicChannelAttention,EnhancedDynamicChannelAttention_update
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print('atten_shape==', p_attn.shape)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn




class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn
        self.head = 8
        self.d_model = 512
        # self.norm = norm

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix):
        # print(f"tgt_shape:{tgt.shape}")
        tgt = fusion_method(tgt, prefer_vector)
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None):
        embeddings = self.tgt_embed(tgt)
        
        # new_model = EnhancedDynamicChannelAttention(2, d_model).to(embeddings.device)
        # embeddings = new_model(embeddings, pref_tensor)
        # Generate key/value features
        

        # embeddings = embeddings + prefer_feature
        batch_size, patch_num, d_model = embeddings.shape
        pref_tensor = torch.tensor([prefer_vector], dtype=torch.float32, device=embeddings.device)
        # 
        attn_prefer = MultiHeadedAttention(self.head, self.d_model).to(embeddings.device)
        #
        preference_expander = PreferenceExpansion(2, d_model, batch_size, patch_num).to(embeddings.device)
        #
        query_input = preference_expander(pref_tensor).to(embeddings.device) * 50
        # # Generate key/value features
        key_input = embeddings
        #
        value_input = embeddings
        #
        # # Multi-head attention
        prefer_feature = attn_prefer(query_input, key_input, value_input)
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0),
                                                                memory_matrix.size(1))
        responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix)
        embeddings = embeddings + 0.5 * prefer_feature
        

        # Multi-head attention
        
        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)



class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory

        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.5, r = 16):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]
        

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # print(f"query shape: {query.shape}")
        # print(f"key shape: {key.shape}")
        # print(f"value shape: {value.shape}")

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class E_PVF_D(AttModel):

    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),cmn)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(E_PVF_D, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk

        tgt_vocab = self.vocab_size + 1
        self.prefer_vector_dim = args.prefer_dim

        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)

        self.model = self.make_model(tgt_vocab, self.cmn)
        self.logit = nn.Linear(args.d_model, tgt_vocab)


        self.memory_matrix = nn.Parameter(torch.FloatTensor(args.cmm_size, args.cmm_dim))

        nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)
        # self.memory_matrix = nn.Parameter(torch.randn((args.cmm_size, args.cmm_dim)), requires_grad=True)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # image_feature_dim = fc_feats.shape[1]
        # # print('shape == ' ,fc_feats.shape)
        #
        # fusion_model = FeatureFusion(image_feature_dim, self.prefer_vector_dim).to('cuda:3')
        # # print('vector == ', prefer_vector)
        # fc_feats = fusion_model(fc_feats, prefer_vector) # 8 * 4096
        # # print('fc_feats == ', fc_feats)
        # # print('shape2 ==== ', fc_feats.shape)
        # fc_feats.to('cuda:3')
        # print('shape ====', fc_feats.shape)
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        # att_fuse_model = FeatureFusionModule().to("cuda:3")
        # prefer_tensor = torch.tensor(prefer_vector).to("cuda:3")
        # att_feats.to("cuda:3")
        # att_feats = att_fuse_model(att_feats, prefer_tensor).to("cuda:3")
        # print('atten_shape==', att_feats.shape)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks) #atten_feats 8 * 98 * 2048
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)


        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        # Memory querying and responding for visual features
        dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0),
                                                                     self.memory_matrix.size(1))
        # att_feats = pack_wrapper(self.norm_vis, att_feats, att_masks)
        responses = self.cmn(att_feats, dummy_memory_matrix, dummy_memory_matrix)
        # att_feats = att_feats + responses
        att_feats = fusion_method(att_feats, prefer_vector).to(att_feats.device)
        # print("1111")
        layer_norm = LayerNorm(self.d_model).to(att_feats.device)
        # att_feats = layer_norm(att_feats).to(att_feats.device)
        # Memory querying and responding for visual features

        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=self.memory_matrix)
        outputs = F.log_softmax(self.logit(out), dim=-1)

        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        
        out, past = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), past=past,
                                      memory_matrix=self.memory_matrix)
        return out[:, -1], [ys.unsqueeze(0)] + past
#15 15 3
class PreferenceExpansion(nn.Module):
    def __init__(self, input_dim, d_model, batch_size, patch_num):
        super(PreferenceExpansion, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 64),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, d_model)
        )
        # self.linear = nn.Linear(input_dim, d_model)
        self.batch_size = batch_size
        self.patch_num = patch_num

    def forward(self, preference_vector):
        expanded_vector = self.linear(preference_vector)
        return expanded_vector.expand(self.batch_size, self.patch_num, -1)
prefer_vector = [1,0]
#50 40 
def fusion_method(embedding, prefer_vector, num_layers = 1):
    batch_size, patch_num, d_model = embedding.shape
    pref_tensor = torch.tensor([prefer_vector], dtype=torch.float32, device=embedding.device)
    # attn_prefer = MultiHeadedAttention(4, 512).to(embedding.device)
    pref_tensor[0][0] *= 100
    pref_tensor[0][1] *= 100
    factor = 3
    factor += prefer_vector[1] * 5
    # pref_tensor[0][2] *= 100
    #
    # factor = 0.5
    #
    # factor += seq_x
    #
    # preference_expander = PreferenceExpansion(2, d_model, batch_size, patch_num).to(embedding.device)
    for _ in range(num_layers):
        attn_prefer = MultiHeadedAttention(4, 512).to(embedding.device)
        preference_expander = PreferenceExpansion(2, d_model, batch_size, patch_num).to(embedding.device) 

        query_input = preference_expander(pref_tensor).to(embedding.device)
        key_input = embedding
        value_input = embedding

        prefer_feature = attn_prefer(query_input, key_input, value_input)
        layer_norm = LayerNorm(512).to(embedding.device)
        embedding = embedding + factor * prefer_feature
        # embedding = layer_norm(embedding)
    # 

    return embedding  