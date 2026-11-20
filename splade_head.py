import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from config import ModelConfig
from model import Shortcake


class SPLADESparseHead(nn.Module):
    """SPLADE Sparse Lexical & Expansion Representation Head.
    
    Transforms token hidden states H (B, L, D) into sparse vocabulary importance weights
    and max-pools across sequence length:
    
      w_{i, j} = log(1 + ReLU(W * h_i + b)_j)
      v_j = max_i (w_{i, j})
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.LayerNorm(config.d_model),
        )
        self.decoder = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: (B, L, D)
        Returns:
            sparse_vector: (B, Vocab) max-pooled non-negative term weights
            flops_loss: scalar sparsity penalty sum(w_j^2)
        """
        h = self.transform(hidden_states) # (B, L, D)
        logits = self.decoder(h) # (B, L, Vocab)

        # SPLADE log-ReLU transformation: log(1 + max(0, logits))
        term_weights = torch.log(1.0 + torch.relu(logits)) # (B, L, Vocab)

        # Max-pooling over sequence length
        sparse_vector, _ = torch.max(term_weights, dim=1) # (B, Vocab)

        # Regularization / FLOPs sparsity loss
        flops_loss = torch.sum(torch.mean(sparse_vector, dim=0) ** 2)

        return sparse_vector, flops_loss


class ShortcakeSPLADE(nn.Module):
    """Shortcake backbone equipped with a SPLADE sparse vector head."""

    def __init__(self, backbone: Shortcake):
        super().__init__()
        self.backbone = backbone
        self.splade_head = SPLADESparseHead(backbone.config)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids)
        h = outputs["hidden_states"]
        sparse_vector, flops_loss = self.splade_head(h)
        return {
            "sparse_vector": sparse_vector,
            "flops_loss": flops_loss,
            "hidden_states": h,
        }
