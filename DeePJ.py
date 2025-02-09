import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from torch_geometric.nn import DenseGraphConv, dense_diff_pool


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embedder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class TimeEncoding(torch.nn.Module):
    def __init__(self, d_model, max_elapsed_time):
        """
        Time Encoding Module.
        Args:
            d_model: The dimension of the model embeddings (must be even).
            max_elapsed_time: Maximum elapsed time in the dataset (minutes).
        """
        super(TimeEncoding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even."
        self.d_model = d_model
        self.max_elapsed_time = max_elapsed_time

    def forward(self, intervals):
        """
        Compute time encoding for a batch of elapsed times.
        Args:
            intervals: A tensor of shape (batch_size, seq_len), containing elapsed times.
        Returns:
            time_encoding: A tensor of shape (batch_size, seq_len, d_model), representing time encodings.
        """
        batch_size, seq_len = intervals.size()
        device = intervals.device
        
        # Precompute the denominator (shape: d_model // 2)
        position = torch.arange(self.d_model // 2, device=device).float()
        denominator = torch.pow(self.max_elapsed_time, 2 * position / self.d_model)

        # Expand intervals for broadcasting (shape: batch_size, seq_len, d_model // 2)
        numerator = intervals.unsqueeze(-1)

        # Compute sine and cosine components
        sinusoidal = numerator / denominator  # Shape: (batch_size, seq_len, d_model // 2)
        time_encoding = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        time_encoding[:, :, 0::2] = torch.sin(sinusoidal)  # Even indices
        time_encoding[:, :, 1::2] = torch.cos(sinusoidal)  # Odd indices

        return time_encoding
    
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model):
        """
        A self-attention layer with optional prior-based attention scores.

        Args:
            d_model (int): Dimension of the model (input features).
            dropout (float): Dropout rate for attention scores.
        """
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.W_query = nn.Linear(d_model, d_model)
        self.W_key = nn.Linear(d_model, d_model)
        self.W_value = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, pad_masks=None, causal_masks=None, prior=None):
        """
        Forward pass for self-attention. If pad_masks, causal_masks and prior are all None, 
        then the attention is a standard self-attention.
        Args:
            query: tensor to generate Query tensor of shape (batch_size, seq_len, d_model).
            key: tensor to generate Key tensor of shape (batch_size, seq_len, d_model).
            value: tensor to generate Value tensor of shape (batch_size, seq_len, d_model).
            pad_masks: Padding masks of shape (batch_size, seq_len).
            causal_masks: Causal masks of shape (batch_size, seq_len, seq_len).
            prior: Prior tensor of shape (batch_size, seq_len, seq_len), optional.
        """
        # Compute Q, K, V
        Q = self.W_query(query)  # (batch_size, seq_len, d_model)
        K = self.W_key(key)      # (batch_size, seq_len, d_model)
        V = self.W_value(value)  # (batch_size, seq_len, d_model)

        # Compute raw attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5) # (batch_size, seq_len, seq_len)
        
        # Apply masks to the attention scores
        if (pad_masks is not None) and (causal_masks is not None):
            # Combine pad_masks and causal_mask
            combined_mask = (pad_masks.unsqueeze(1) & causal_masks).bool()

            # Apply the combined mask to the scores
            scores = scores.masked_fill(~combined_mask, float('-inf'))

            # Handle fully masked rows by setting them to 0
            scores = torch.where(combined_mask.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        if (pad_masks is not None) and (causal_masks is not None):
            # if one row is fully masked, set the attention weights to 0
            attn_weights = torch.where(combined_mask.any(dim=-1, keepdim=True), attn_weights, torch.zeros_like(attn_weights))
            # further mask attention weights with pad_masks vertically
            attn_weights = attn_weights * (pad_masks.unsqueeze(1).expand(-1, pad_masks.size(1), -1).transpose(-1, -2))

        # Compute final output, if prior is not None, use prior to replace the attention weights
        if prior is not None:
            seq_emds = torch.matmul(prior, V) # (batch_size, seq_len, d_model)
        else:
            seq_emds = torch.matmul(attn_weights, V) # (batch_size, seq_len, d_model)

        # free up memory
        del Q, K, V
        return seq_emds, attn_weights
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model, 1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_emds, sublayer, sublayer_name):
        # sublayer should be a function that takes seq_emds as input and returns a transformed seq_emds
        "Apply residual connection to any sublayer with the same size."
        if sublayer_name == 'self-attention':
            seq_emds_new, attn_weights = sublayer(self.norm(seq_emds))
            seq_emds_new = self.dropout(seq_emds_new)
            residual = seq_emds + seq_emds_new
            return residual, attn_weights
        
        elif sublayer_name == 'feed-forward':
            seq_emds_new = sublayer(self.norm(seq_emds))
            seq_emds_new = self.dropout(seq_emds_new)
            residual = seq_emds + seq_emds_new
            return residual
        
class GCTLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super(GCTLayer, self).__init__()
        self.self_attn = SelfAttentionLayer(d_model)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        
    def forward(self, seq_emds, pad_masks, causal_masks, priors):
        # lambda x: self.self_attn(x, x, x, mask) explained:
        # self.self_attn is a class that takes 4 arguments: query, key, value, mask, and it has been initialized as an object
        # since the self_attn object is callable (with forward), we can pass the input x to it by calling it as a function
        # here we use lambda to create a function because we have other arguments "mask", to pass to the self_attn object
        # so for feed_forward, we can just pass it as an argument to the sublayer
        seq_emds, attn_weights = self.sublayer[0](seq_emds, lambda seq_emds: self.self_attn(seq_emds, seq_emds, seq_emds, 
                                                                              pad_masks, causal_masks, priors), 'self-attention')
        seq_emds = self.sublayer[1](seq_emds, self.feed_forward, 'feed-forward')
        return seq_emds, attn_weights
    
class GraphConvTransformer(nn.Module):
    def __init__(self, d_model, num_GCT_layers, dropout):
        super(GraphConvTransformer, self).__init__()
        self.d_model = d_model
        self.num_GCT_layers = num_GCT_layers
        self.GCT_layers = nn.ModuleList([GCTLayer(d_model, dropout) for _ in range(num_GCT_layers)])
        
    def forward(self, seq_emds, pad_masks, causal_masks, priors):
        attn_matrices_layers = [priors]
        for i in range(self.num_GCT_layers):
            # we don't want to use the prior matrices for layers after the first layer
            if i == 0: 
                prior = priors
            else:
                prior = None    
            seq_emds, attn_weights = self.GCT_layers[i](seq_emds, pad_masks, causal_masks, prior)
            attn_matrices_layers.append(attn_weights)
        KLD_loss = self.get_KLD_loss(attn_matrices_layers)
        # we use the last attention matrix as the ajacent matrix
        return seq_emds, attn_matrices_layers[-1], KLD_loss
    
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
    
class GNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GNN, self).__init__()
        # DenseSAGEConv expects to work on binary adjacency matrices.
        # If you want to make use of weighted dense adjacency matrices, 
        # please use torch_geometric.nn.dense.DenseGraphConv instead.
        self.conv = DenseGraphConv(in_channel, out_channel)
        self.batch_norm = torch.nn.BatchNorm1d(out_channel) # expect (N,C,L), where N is the 
        # batch size, C is the number of channels, and L is the seq len / number of nodes.
        
        
    def forward(self, node_emds, adj_matrices, pad_masks, residual_connection):
        if residual_connection:
            node_emds = node_emds + self.conv(node_emds, adj_matrices, pad_masks)
            # (batch_size, num_nodes, d_model)
        else:
            node_emds = self.conv(node_emds, adj_matrices, pad_masks)
        node_emds = self.batch_norm(node_emds.transpose(1, 2)).transpose(1, 2)
        return node_emds.relu()
    
class DiffPool(nn.Module):
    def __init__(self, d_model, num_clusters, dropout):
        super(DiffPool, self).__init__()
        self.d_model = d_model
        self.num_clusters = num_clusters
        self.gnn_pool = GNN(d_model, num_clusters)
        self.gnn_embed = GNN(d_model, d_model)
        self.batch_norm = torch.nn.BatchNorm1d(d_model) # expect (N,C,L), where N is the 
        # batch size, C is the number of channels, and L is the seq len / number of nodes.
        self.dropout = torch.nn.Dropout(dropout)
        
        
    def forward(self, node_emds, adj_matrices, pad_masks):
        s = self.gnn_pool(node_emds, adj_matrices, pad_masks, residual_connection = False)
        node_emds = self.gnn_embed(node_emds, adj_matrices, pad_masks, residual_connection = True)
        node_emds = self.dropout(node_emds)
        # For s, the softmax does not have to be applied before-hand, since it is executed within this method.
        embed_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(node_emds, adj_matrices, s, pad_masks, normalize = True)
        # soft cluster assignment
        cluster_assign = torch.softmax(s, dim=-1)
        embed_pool = self.batch_norm(embed_pool.transpose(1, 2)).transpose(1, 2)
        embed_pool = embed_pool.relu()
        embed_pool = self.dropout(embed_pool)
        return embed_pool, adj_pool, cluster_assign, link_loss, ent_loss
    
class GlobalPool(nn.Module):
    def __init__(self, d_model):
        """
        Aggregates input embeddings into a single global vector using attention pooling.
        :param d_model: Dimensionality of the input embeddings.
        """
        super(GlobalPool, self).__init__()
        # Learnable weight to compute attention scores
        self.attn_weight = nn.Linear(d_model, 1)
    
    def forward(self, node_emds):
        """
        Args:
            node_emds: Input tensor of shape [batch_size, num_clusters, d_model]
        
        Returns:
            output: Aggregated vector of shape [batch_size, d_model]
            attn_weights: Attention weights of shape [batch_size, num_clusters]
        """
        # Compute attention scores for each vector
        attn_scores = self.attn_weight(node_emds).squeeze(-1)  # Shape: [batch_size, num_clusters]
        attn_weights = F.softmax(attn_scores, dim=-1)  # Normalize scores across num_clusters vectors
        
        # Weighted sum of input vectors
        output = torch.bmm(attn_weights.unsqueeze(1), node_emds).squeeze(1)  # Shape: [batch_size, d_model]
        
        return output, attn_weights
    
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_linear_layers=1):
        """
        Args:
            input_dim (int): The number of input features.
            num_classes (int): The number of output classes.
        """
        super(LinearClassifier, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_linear_layers - 1)] 
            + [nn.Linear(input_dim, num_classes)]
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)  # LogSoftmax activation

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            log_probs: Log probabilities of shape [batch_size, num_classes]
        """
        if len(self.linears) > 1:
            for linear in self.linears[:-1]:
                x = F.relu(linear(x))  # ReLU activation
        logits = self.linears[-1](x)  # Call the last layer explicitly
        log_probs = self.log_softmax(logits)  # Apply LogSoftmax
        return log_probs
    
class DeepJourney(nn.Module):
    def __init__(self, embedder, time_encoder, structure_learner, graph_pooler, global_pooler, classifier):
        super(DeepJourney, self).__init__()
        self.embedder = embedder # encode the input sequence with discrete ints to dense vectors
        self.time_encoder = time_encoder # provide the mpdel with the time information of each encounter
        self.structure_learner = structure_learner # transformer layers that learn the graph structure
        self.graph_pooler = graph_pooler # Diff Pool layer that cluter the graph into subgraphs
        self.global_pooler = global_pooler # simple attention that pools the graph into a single vector
        self.classifier = classifier # classifier that makes the final prediction
        
    def embed(self, code_ints):
        """
        Embed the input sequence and add time encodings.
        Args:
            code_ints: input sequence of shape (batch_size, seq_len)
            intervals: elapsed time of each encounter of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Embedded sequence of shape (batch_size, seq_len, d_model).
        """
        return self.embedder(code_ints)
    
    def time_encode(self, intervals):
        """
        Add time encodings to the input sequence.
        Args:
            intervals: elapsed time of each encounter of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Time-encoded sequence of shape (batch_size, seq_len, d_model).    
        """
        return self.time_encoder(intervals)
    
    def learn_structure(self, seq_emds, pad_masks, causal_masks, prior_matrices):
        """
        Args:
            seq_emds (torch.Tensor): Sequence embeddings of shape (batch_size, seq_len, d_model).
            pad_masks (torch.Tensor): Padding mask of shape (batch_size, seq_len).
            causal_masks (torch.Tensor): Causal mask of shape (batch_size, seq_len, seq_len).
            prior_matrices (torch.Tensor): Prior matrices of shape (batch_size, seq_len, seq_len).
        
        Returns:
            tuple: Node embeddings, adjacency matrices, and KLD loss.
        """
        node_emds, adj_matrices, KLD_loss = self.structure_learner(seq_emds, pad_masks, causal_masks, prior_matrices)
        return node_emds, adj_matrices, KLD_loss
    
    def pool_graph(self, node_emds, adj_matrices, pad_masks):
        """
        Pool the graph embeddings into subgraphs and return PyG-compatible outputs.

        Args:
            node_emds (torch.Tensor): Node embeddings of shape (batch_size, seq_len, d_model).
            pad_masks (torch.Tensor): Padding mask of shape (batch_size, seq_len).
            adj_matrices (torch.Tensor): Adjacency matrices of shape (batch_size, seq_len, seq_len).

        Returns:
            tuple: Updated node embeddings, edge indices, and edge weights.
        """
        embed_pool, adj_pool, cluster_assign, link_loss, ent_loss = self.graph_pooler(node_emds, adj_matrices, pad_masks)
        return embed_pool, adj_pool, cluster_assign, link_loss, ent_loss
    
    def embed_graph(self, node_emds):
        """
        Pool the entire graph into a single embedding vector.
        
        Args:
            node_emds (torch.Tensor): Node embeddings of shape (batch_size, num_clusrers, d_model).
        
        Returns:
            torch.Tensor: Pooled graph embedding of shape (batch_size, d_model).
        """
        graph_embed, cluster_weights = self.global_pooler(node_emds)
        return graph_embed, cluster_weights
    
    def classify(self, graph_emds):
        """
        Perform classification based on the graph embeddings.
        
        Args:
            graph_emds (torch.Tensor): Graph embeddings of shape (batch_size, d_model).
        
        Returns:
            torch.Tensor: Classification logits (log_softmax).
        """
        return self.classifier(graph_emds)
    
    def forward(self, code_ints, pad_masks, causal_masks, enc_intervals, prior_matrices):
        # embed the input sequence from discrete ints to dense vectors, add time encodings
        seq_emds = self.embed(code_ints) + self.time_encode(enc_intervals)
        
        # learn the latent graph structure by masked self-attention
        node_emds, adj_matrices, KLD_loss = self.learn_structure(seq_emds, pad_masks, causal_masks, prior_matrices)
        
        # cluster the graph into subgraphs and pool the subgraphs into a single vector
        embed_pool, _, cluster_assign, link_loss, ent_loss = self.pool_graph(node_emds, adj_matrices, pad_masks)
        
        # pool the graph into a single vector
        graph_embed, cluster_weights = self.embed_graph(embed_pool)
        
        # make the final prediction
        logits = self.classify(graph_embed)
        
        return logits, adj_matrices, cluster_assign, cluster_weights, KLD_loss, link_loss, ent_loss
    
def make_deepj_model(d_model, num_GCT_layers, num_linear_layers, 
                     dropout, vocab_size, max_elapsed_time, num_clusters, 
                     num_classes):
    embedder = Embedder(d_model, vocab_size)
    time_encoder = TimeEncoding(d_model, max_elapsed_time)
    structure_learner = GraphConvTransformer(d_model, num_GCT_layers, dropout)
    graph_pooler = DiffPool(d_model, num_clusters, dropout)
    global_pooler = GlobalPool(d_model)
    classifier = LinearClassifier(d_model, num_classes, num_linear_layers)
    model = DeepJourney(embedder, time_encoder, structure_learner, graph_pooler, global_pooler, classifier)
    return model