import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_model_dim, max_seq_len) -> None:
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_model_dim
        self.max_seq_len = max_seq_len
        self.pe = self.create_positional_encoding()
    
    def get_angles(self, pos, k, d):
        i = k // 2
        angles = pos / np.power(10000, 2 * i / d)
        return angles
    
    def create_positional_encoding(self):

        pos = np.arange(self.max_seq_len)[:, np.newaxis]
        k = np.arange(self.embed_dim)[np.newaxis, :]
        d = self.embed_dim

        angle_rads = self.get_angles(pos, k, d)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pe = angle_rads
        
        return torch.from_numpy(pe).float().to(DEVICE)
    
    def forward(self, x):
        """
        Add the positional encoding with the token embedding
        """
        return x + self.pe[:x.size(1), :]


class MaskedAttention(torch.nn.Module):
    def __init__(self, embed_dim, head_dim, block_size: int) -> None:
        super(MaskedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        # key, query, and value matrices
        self.query_layer = torch.nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.key_layer = torch.nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.value_layer = torch.nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        # self.out = torch.nn.Linear(self.n_heads * self.head_dim, self.embed_dim)
    
    def forward(self, x):
        B, T, C = x.shape

        k = self.key_layer(x)
        q = self.query_layer(x)
        v = self.value_layer(x)

        matmul_q = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.head_dim)

        # mask = mask.reshape(matmul_q.shape[0], 1, matmul_q.shape[2])
        matmul_q = matmul_q.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        
        scores = self.softmax(matmul_q)
        output = torch.bmm(scores, v)
        return output


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, n_heads, block_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads

        self.self_attn_layers = torch.nn.ModuleList([MaskedAttention(embed_dim, self.head_dim, block_size)
                                                     for _ in range(n_heads)]
        )
        self.out = torch.nn.Linear(n_heads * self.head_dim, embed_dim)
    
    def forward(self, x):
        attn_outs = [self_attn(x) for self_attn in self.self_attn_layers]
        concat_attn_outs = torch.cat(attn_outs, dim=-1)
        return self.out(concat_attn_outs)


class FullyConnected(torch.nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4 * embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * embed_dim, embed_dim)
        )
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x):
        return self.net(x)
    

class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, n_heads, block_size, dropout_rate) -> None:
        super().__init__()
        self.mha1 = MultiHeadAttention(embed_dim, n_heads, block_size)
        self.ffn_1 = FullyConnected(embed_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layernorm1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm2 = torch.nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        norm_x = self.layernorm1(x)
        attn_out = self.mha1(norm_x)
        res_out = x + attn_out
        norm_res_out = self.layernorm2(res_out)

        ffn_out = self.ffn_1(norm_res_out)
        ffn_out = self.dropout(ffn_out)
        
        return res_out + ffn_out
    

class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, n_layers, n_heads,
                 block_size, vocab_size, dropout_rate=0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.block_size = block_size

        self.decoder_stack = torch.nn.Sequential(
            *[
                DecoderLayer(
                    embed_dim=self.embed_dim,
                    n_heads=self.n_heads,
                    block_size=self.block_size,
                    dropout_rate=self.dropout_rate
                )
                for _ in range(self.n_layers)
            ]
        )

        self.token_embedding_tbl = torch.nn.Embedding(self.vocab_size, self.embed_dim)
        # self.positional_encoding_tbl = torch.nn.Embedding(self.block_size, self.embed_dim)
        self.positional_encoding_tbl = PositionalEncoding(self.embed_dim, self.block_size)
        self.layernorm1 = torch.nn.LayerNorm(self.embed_dim)
        self.proj_out = torch.nn.Linear(self.embed_dim, self.vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding_tbl(idx)
        # positional_embedding = self.positional_encoding_tbl(torch.arange(T, device=DEVICE))
        
        x = self.positional_encoding_tbl(token_embedding)

        # x = token_embedding + positional_embedding
        x = self.decoder_stack(x)
        x = self.layernorm1(x)
        logits = self.proj_out(x)   # output shape: (batch_size, sequence_length, vocab_size)
        
        if targets is not None:
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            targets = torch.reshape(targets, (B * T,))
            loss = torch.nn.functional.cross_entropy(logits, targets)
        else:
            loss = None
        
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx