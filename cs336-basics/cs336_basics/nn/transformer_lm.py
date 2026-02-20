import torch
from torch import nn
import einx
from cs336_basics.nn.embedding import Embedding
from cs336_basics.nn.transformer_block import TransformerBlock
from cs336_basics.nn.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    """基于 TransformerBlock 的语言模型

    结构：token embedding → N×(RMSNorm + MHA + 残差，RMSNorm + SwiGLU FFN + 残差) → RMSNorm → LM Head
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        ffn_style: str = "swiglu",
        ffn_match_params: bool = False,
        dropout_p: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        self.token_embeddings = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    use_rmsnorm=use_rmsnorm,
                    use_rope=use_rope,
                    norm_style=norm_style,
                    ffn_style=ffn_style,
                    ffn_match_params=ffn_match_params,
                    dropout_p=dropout_p,
                    max_seq_len=self.context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=self.d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.lm_head = nn.Parameter(torch.empty((self.vocab_size, self.d_model), device=device, dtype=dtype))

        torch.nn.init.trunc_normal_(self.lm_head, mean=0.0, std=(2.0 / (self.vocab_size + self.d_model)) ** 0.5, a=-0.5, b=0.5)

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        """前向：输入为 token id，输出为每个位置的词表 logits"""
        x = self.token_embeddings(x_idx)
        b = x.shape[0]
        t = x.shape[1]
        pos = torch.arange(t, device=x.device, dtype=torch.long)
        ones = torch.ones((b,), device=x.device, dtype=torch.long)
        token_positions = einx.multiply("b, t -> b t", ones, pos)
        h = x
        for layer in self.layers:
            h = layer(h, token_positions=token_positions)
        h = self.ln_final(h)
        logits = torch.einsum("... t d, v d -> ... t v", h, self.lm_head)
        return logits
