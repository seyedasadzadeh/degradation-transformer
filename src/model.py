import torch

class TokenPositionEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, context_window, embedding_dim):
        super().__init__()
        self.token_embed = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = torch.nn.Embedding(context_window, embedding_dim)
    def forward(self, x):
        # embedding will be added as the last dimention
        tex = self.token_embed(x)
        pex = self.pos_embed(torch.arange(x.size(-1), device=x.device).unsqueeze(0))
        return tex+pex


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        
        # Q, K, V projections
        self.qproject = torch.nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.kproject = torch.nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.vproject = torch.nn.Linear(embedding_dim,embedding_dim,bias=False)
        # Output projection
        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads
        self.out_project = torch.nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.layernorm = torch.nn.LayerNorm(embedding_dim)
        
    def forward(self,x):
        from math import sqrt
        #remember input is 32,40,128
        q=self.qproject(x)
        k=self.kproject(x)
        v=self.vproject(x)

        # reshape to heads
        q = torch.reshape(q, (q.size(0),-1,self.num_heads, self.head_dim)).permute(0,2,1,3) # 32,8,40,16
        k = torch.reshape(k, (k.size(0),-1,self.num_heads, self.head_dim)).permute(0,2,1,3) # 32,8,40,16
        v = torch.reshape(v, (v.size(0),-1,self.num_heads, self.head_dim)).permute(0,2,1,3) # 32,8,40,16

        score = q@k.permute(0,1,3,2) / sqrt(self.head_dim) # 32,8,40,40
        soft_score = torch.softmax(score, dim=-1) # 32,8,40,40
        scoreV = soft_score@v  # 32,8,40,16
        sv = scoreV.permute(0,2,1,3).contiguous().view(scoreV.size(0), scoreV.size(2), -1)        
        mha_out = self.out_project(sv) #32,40,128
        mha_out_plus_input = mha_out + x #32,40,128
        out = self.layernorm(mha_out_plus_input) #32,40,128
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.first_proj = torch.nn.Linear(embedding_dim, int(4*embedding_dim))
        self.relu = torch.nn.ReLU()
        self.second_proj = torch.nn.Linear(int(4*embedding_dim), embedding_dim)
        self.layernorm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x is shape 32, 40, 128
        x_proj1 = self.first_proj(x) # 32, 40, 4*128
        x_proj1_relu = self.relu(x_proj1)# 32, 40, 4*128
        x_proj2 = self.second_proj(x_proj1_relu)# 32, 40, 128
        x_proj2_plus_input = x_proj2 + x # 32, 40, 4*128
        out = self.layernorm(x_proj2_plus_input) # 32, 40, 4*128
        return out

class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        # Add your MultiHeadAttention and FeedForward here
        self.mha = MultiHeadAttention(embedding_dim, num_heads)
        self.ff = FeedForward(embedding_dim)

    def forward(self, x):
        mha_out = self.mha(x)
        out = self.ff(mha_out)
        return out

class DegradationTransformer(torch.nn.Module):
    def __init__(self, vocab_size, context_window, embedding_dim, num_heads, num_blocks, metadata_dim=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.metadata_dim = int(metadata_dim)
        self.tpembed = TokenPositionEmbedding(vocab_size, context_window, embedding_dim)
        if self.metadata_dim > 0:
            self.metadata_projector = torch.nn.Sequential(
                torch.nn.LayerNorm(self.metadata_dim),
                torch.nn.Linear(self.metadata_dim, embedding_dim),
            )
        else:
            self.metadata_projector = None
        self.tbls_list = torch.nn.ModuleList([TransformerBlock(embedding_dim, num_heads) for _ in range(num_blocks)])
        self.lm_head = torch.nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tpembed.token_embed.weight  # Weight tying

    def forward(self, x, metadata=None):
        # input is 32*40
        x = self.tpembed(x) # 32*40*128
        if self.metadata_projector is not None:
            if metadata is None:
                metadata = torch.zeros(x.size(0), self.metadata_dim, device=x.device, dtype=x.dtype)
            else:
                metadata = metadata.to(device=x.device, dtype=x.dtype)
            metadata_emb = self.metadata_projector(metadata).unsqueeze(1)
            x = x + metadata_emb
        for block in self.tbls_list:
            x = block(x) # 32*40*128
        last_x = x[:,-1, :] # 32*128
        vocab_x = self.lm_head(last_x) # 32*300
        return vocab_x

# export learner.py
