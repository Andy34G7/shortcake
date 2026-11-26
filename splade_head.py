import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from config import ModelConfig
from model import Shortcake, RMSNorm


class SPLADESparseHead(nn.Module):
    """SPLADE Sparse Lexical & Expansion Representation Head.

    Transforms token hidden states H (B, L, D) into sparse non-negative vocabulary
    weights and max-pools across non-pad sequence positions:

      w_{i, j} = log(1 + ReLU(W * h_i + b)_j)
      v_j = max_{i in valid_tokens} (w_{i, j})

    When `lm_head_weight` is provided, ties the vocabulary decoder to the pre-trained
    LM head and initializes the residual transform to zero so step-0 lexical alignment
    from pre-training is preserved.
    """

    def __init__(
        self,
        config: ModelConfig,
        lm_head_weight: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=True)

        # Initialize residual adapter to zero so initial logits match pre-trained lm_head
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.decoder.bias)

        if lm_head_weight is not None:
            self.decoder.weight = lm_head_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: (B, L, D)
        attention_mask: optional (B, L) boolean/binary mask (1 = real token, 0 = pad)
        Returns:
            sparse_vector: (B, Vocab) max-pooled non-negative term weights
            flops_loss: scalar FLOPs sparsity penalty sum_j(mean_i(w_{i,j})^2)
        """
        h = self.norm(hidden_states + self.proj(F.silu(hidden_states)))  # (B, L, D)
        logits = self.decoder(h)  # (B, L, Vocab)

        # SPLADE log-ReLU transformation: log(1 + ReLU(logits))
        term_weights = torch.log1p(F.relu(logits))  # (B, L, Vocab)

        # Zero out padding positions before max-pooling so <pad> never leaks into sparse_vector
        if attention_mask is not None:
            mask = attention_mask.to(dtype=term_weights.dtype).unsqueeze(-1)  # (B, L, 1)
            term_weights = term_weights * mask

        # Max-pooling over sequence length
        sparse_vector, _ = torch.max(term_weights, dim=1)  # (B, Vocab)

        # FLOPs regularization loss (computed in float32 for stability)
        flops_loss = torch.sum(torch.mean(sparse_vector.float(), dim=0) ** 2)

        return sparse_vector, flops_loss


class ShortcakeSPLADE(nn.Module):
    """Shortcake backbone equipped with a pre-trained-tied SPLADE sparse vector head."""

    def __init__(
        self,
        backbone: Shortcake,
        tie_lm_head: bool = True,
        pad_token_id: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.pad_token_id = pad_token_id
        self.bidirectional = bidirectional
        lm_weight = backbone.lm_head.weight if tie_lm_head else None
        self.splade_head = SPLADESparseHead(backbone.config, lm_head_weight=lm_weight)
        if bidirectional:
            from model import MambaBlockWrapper
            n_mamba = sum(1 for layer in backbone.layers if isinstance(layer, MambaBlockWrapper))
            # Zero-initialized residual gate per Mamba layer preserves pretrained causal reps at step 0
            self.bwd_gates = nn.ParameterList(
                [nn.Parameter(torch.zeros(backbone.config.d_model)) for _ in range(n_mamba)]
            )
        else:
            self.bwd_gates = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)

        # Skip redundant backbone.lm_head matmul since SPLADE head computes vocabulary logits
        outputs = self.backbone(
            input_ids,
            return_logits=False,
            bidirectional=self.bidirectional,
            attention_mask=attention_mask,
            bwd_gates=self.bwd_gates,
        )
        h = outputs["hidden_states"]
        sparse_vector, flops_loss = self.splade_head(h, attention_mask=attention_mask)
        return {
            "sparse_vector": sparse_vector,
            "flops_loss": flops_loss,
            "hidden_states": h,
        }
