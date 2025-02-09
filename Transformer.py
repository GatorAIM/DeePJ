from DeePJ import GCTLayer
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, dropout, num_layers):
        super(Transformer, self).__init__()
        self.attn_layers = nn.ModuleList([GCTLayer(d_model, dropout) for _ in range(num_layers)])
        
    def forward(self, seq_emds, pad_masks, causal_masks):
        for layer in self.attn_layers:
            seq_emds, _ = layer(seq_emds, pad_masks, causal_masks, priors = None)
        # seq_emds: [batch_size, seq_len, d_model]
        readout = torch.mean(seq_emds, dim=1)
        return readout
