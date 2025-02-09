from DeePJ import GCTLayer
import torch
import torch.nn as nn

class GCT(nn.Module):
    def __init__(self, d_model, dropout, num_layers):
        super(GCT, self).__init__()
        self.GCT_layers = nn.ModuleList([GCTLayer(d_model, dropout) for _ in range(num_layers)])
        # self.pooler = Pooler(d_model)
    def forward(self, seq_emds, pad_masks, causal_masks, priors):
        attn_matrices_layers = [priors]
        for i in range(len(self.GCT_layers)):
            # we don't want to use the prior matrices for layers after the first layer
            if i == 0: 
                prior = priors
            else:
                prior = None    
            seq_emds, attn_weights = self.GCT_layers[i](seq_emds, pad_masks, causal_masks, prior)
            attn_matrices_layers.append(attn_weights)
        KLD_loss = self.get_KLD_loss(attn_matrices_layers)
        readout = torch.mean(seq_emds, dim=1)
        return readout, KLD_loss
    
    @staticmethod
    def get_KLD_loss(attn_matrices):
        kl_terms = []  # To store KL divergence for each pair of consecutive matrices
        for i in range(1, len(attn_matrices)):
            # Retrieve attention matrices
            p = attn_matrices[i-1]  # Shape: [batch_size, seq_len, seq_len]
            q = attn_matrices[i]    # Shape: [batch_size, seq_len, seq_len]

            # Clamp values to prevent numerical instability (e.g., log(0))
            log_p = torch.log(torch.clamp(p, min=1e-12))  # Shape: [batch_size, seq_len, seq_len]
            log_q = torch.log(torch.clamp(q, min=1e-12))  # Shape: [batch_size, seq_len, seq_len]

            # Compute element-wise KL divergence for each position in the sequence
            kl_term = p * (log_p - log_q)  # Element-wise product, Shape: [batch_size, seq_len, seq_len]

            # Sum over the last axis (seq_len) to collapse the second sequence dimension
            kl_term = torch.sum(kl_term, dim=-1)  # Shape: [batch_size, seq_len]

            # Average across the batch (and remaining sequence dimension, seq_len)
            kl_term = torch.mean(kl_term)  # Scalar: Single value per matrix pair

            # Append the computed KL term for this pair
            kl_terms.append(kl_term)

        # Stack KL terms and compute the overall mean
        reg_term = torch.mean(torch.stack(kl_terms))  # Scalar: Final KL loss for all matrix pairs
        return reg_term
        

# class Pooler(nn.Module):
#     def __init__(self, d_model):
#         super(Pooler, self).__init__()
#         self.dense = nn.Linear(d_model, d_model)
#         self.activation = nn.ReLU()
    
#     def forward(self, seq_emds):
#         first_token_tensor = seq_emds[:,0,:] # (batch_size, d_model)
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output