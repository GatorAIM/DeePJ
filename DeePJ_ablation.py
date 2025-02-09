import torch
import torch.nn as nn
from DeePJ import GCTLayer
from DeePJ import Embedder, TimeEncoding, GraphConvTransformer, DiffPool, GlobalPool, LinearClassifier


class GraphConvTransformerAblation(nn.Module):
    def __init__(self, d_model, num_GCT_layers, dropout):
        super(GraphConvTransformerAblation, self).__init__()
        self.d_model = d_model
        self.num_GCT_layers = num_GCT_layers
        self.GCT_layers = nn.ModuleList([GCTLayer(d_model, dropout) for _ in range(num_GCT_layers)])
        
    def forward(self, seq_emds, pad_masks, causal_masks, priors):
        for i in range(self.num_GCT_layers):
            # we don't want to use the prior matrices for layers after the first layer  
            seq_emds, _ = self.GCT_layers[i](seq_emds, pad_masks, causal_masks, priors)
        # we use the last attention matrix as the ajacent matrix
        return seq_emds, priors


class DeepJourneyAblation(nn.Module):
    def __init__(self, embedder, time_encoder, structure_learner, graph_pooler, global_pooler, classifier,
                 use_TE = True, use_SL = True, use_GP = True):
        super(DeepJourneyAblation, self).__init__()

        self.embedder = embedder # encode the input sequence with discrete ints to dense vectors
        self.time_encoder = time_encoder # provide the mpdel with the time information of each encounter
        self.structure_learner = structure_learner # transformer layers that learn the graph structure
        self.graph_pooler = graph_pooler # Diff Pool layer that cluter the graph into subgraphs
        self.global_pooler = global_pooler # simple attention that pools the graph into a single vector
        self.classifier = classifier # classifier that makes the final prediction
        self.use_TE = use_TE # whether to use time encoding
        self.use_SL = use_SL # whether to use structure learning
        self.use_GP = use_GP # whether to use graph pooling
        
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
        if self.use_SL:
            node_emds, adj_matrices, KLD_loss = self.structure_learner(seq_emds, pad_masks, causal_masks, prior_matrices)
            return node_emds, adj_matrices, KLD_loss
        else:
            node_emds, adj_matrices = self.structure_learner(seq_emds, pad_masks, causal_masks, prior_matrices)
            return node_emds, adj_matrices
    
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
        if self.use_TE:
            seq_emds = self.embed(code_ints) + self.time_encode(enc_intervals)
        else:
            seq_emds = self.embed(code_ints)
        
        # learn the latent graph structure by masked self-attention
        if self.use_SL:
            node_emds, adj_matrices, KLD_loss = self.learn_structure(seq_emds, pad_masks, causal_masks, prior_matrices)
        else:
            node_emds, adj_matrices = self.learn_structure(seq_emds, pad_masks, causal_masks, prior_matrices)
        
        # cluster the graph into subgraphs and pool the subgraphs into a single vector
        if self.use_GP:
            embed_pool, _, cluster_assign, link_loss, ent_loss = self.pool_graph(node_emds, adj_matrices, pad_masks)
            # pool the graph into a single vector
            graph_embed, cluster_weights = self.embed_graph(embed_pool)
        else:
            graph_embed = torch.mean(node_emds, dim=1)
        
        # make the final prediction
        logits = self.classify(graph_embed)
        
        if self.use_SL and self.use_GP:
            return logits, adj_matrices, cluster_assign, cluster_weights, KLD_loss, link_loss, ent_loss
        elif not self.use_SL and self.use_GP:
            return logits, adj_matrices, cluster_assign, cluster_weights, link_loss, ent_loss
        elif self.use_SL and not self.use_GP:
            return logits, adj_matrices, KLD_loss
        
def make_deepj_ablation_model(d_model, num_GCT_layers, num_linear_layers, 
                     dropout, vocab_size, max_elapsed_time, num_clusters, 
                     num_classes, use_TE, use_SL, use_GP):
    # assert only one in use_TE, use_SL and use_GP can be False
    assert sum([use_TE, use_SL, use_GP]) >= 2, "At least two of use_TE, use_SL, and use_GP should be True."
    embedder = Embedder(d_model, vocab_size)
    time_encoder = TimeEncoding(d_model, max_elapsed_time)
    if use_SL:
        structure_learner = GraphConvTransformer(d_model, num_GCT_layers, dropout)
    else:
        structure_learner = GraphConvTransformerAblation(d_model, num_GCT_layers, dropout)
    graph_pooler = DiffPool(d_model, num_clusters, dropout)
    global_pooler = GlobalPool(d_model)
    classifier = LinearClassifier(d_model, num_classes, num_linear_layers)
    ablation_model = DeepJourneyAblation(embedder, time_encoder, structure_learner, 
                                         graph_pooler, global_pooler, classifier,
                                         use_TE, use_SL, use_GP)
    return ablation_model