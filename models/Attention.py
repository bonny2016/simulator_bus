import torch
import torch.nn as nn

torch.set_printoptions(precision=2, sci_mode=False)


# key and value (stop features): (N_STOP, N_STOP_FEATURES)
# query (bus features): (N_BUSES, N_BUS_FEATURES)
# dim_q : N_BUS_FEATURES
# dim_k: N_N_STOP_FEATURES
class Attention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_out, proj_d):
        super(Attention, self).__init__()
        self.dim_out = dim_out
        self.proj_d = proj_d
        self.dim_k = dim_k
        self.dim_v = dim_k
        self.dim_q = dim_q
        self.keys = nn.Linear(dim_k, proj_d, bias=False)
        self.queries = nn.Linear(dim_q, proj_d, bias=False)
        self.fc_out = nn.Linear(dim_k, dim_out)
        self.norm1 = nn.LayerNorm(dim_out + dim_q)

    def forward(self, values, keys, queries):
        # project into some other space, so that they can be compared with.
        # values = self.values(values)  # (N, value_len, heads, head_dim)
        keys_projected = self.keys(keys)  # (N, key_len, dim_k) -> (N, key_len, proj_d)
        queries_projected = self.queries(queries)  # (N, query_len, dim_q) -> (N, query_len, proj_d)

        # Einsum does matrix mult.
        # energy: (N, query_len, dim_k)
        energy = torch.einsum("nqd,nkd->nqk", [queries_projected, keys_projected])

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for better stability
        # attention shape: (N, query_len, dim_k)
        attention = torch.softmax(energy / (self.proj_d ** (1 / 2)), dim=2)

        # out after matrix multiply: (N, query_len, dim_k)
        out = torch.einsum("nqk,nkh->nqh", [attention, values])

        # we then reshape the last dimension.
        # final out shape: (N, query_len, dim_out)
        out = self.fc_out(out)
        combined = torch.cat((out, queries), dim=2)
        result = self.norm1(combined)
        return result


class TransformerBlock(nn.Module):
    # steps, heads, head_dim = shape
    def __init__(self, head_dim_k, head_dim_q, embed_d, proj_d):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(head_dim_q, head_dim_k, embed_d, proj_d)
        self.norm1 = nn.LayerNorm(embed_d+head_dim_q)
        self.norm2 = nn.LayerNorm(embed_d+head_dim_q)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_d+head_dim_q, 2 * (embed_d+head_dim_q)),
            nn.ReLU(),
            nn.Linear(2 * (embed_d+head_dim_q), embed_d+head_dim_q),
        )
    # value: x_stops (1x19x4)
    # key: x_stops (1x19x4)
    # query: x_buses (1x6x8)
    def forward(self, value, key, query):
        shape = query.shape
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        # x = self.norm1(attention + query.view(shape[0], -1))
        x = torch.cat((attention, query), dim=2)
        x = self.norm1(x)

        forward = self.feed_forward(x)
        out = self.norm2(forward)
        return out

