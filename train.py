import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from torch_geometric.nn import DenseGraphConv, DenseGCNConv
from pyhealth.models import (RNNLayer, DeeprLayer)
from copy import deepcopy as c
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from scipy.stats import sem, t
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, roc_auc_score, 
                             average_precision_score)
import importlib
import DeePJ
import DeePJ_ablation
import Transformer
import GCT
importlib.reload(DeePJ)
importlib.reload(DeePJ_ablation)
importlib.reload(Transformer)
importlib.reload(GCT)


class Embedder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
    
class GNN(nn.Module):
    def __init__(self, gnn_type, d_model, num_gnn_layers, dropout):
        """
        Args:
            gnn_type: str, either 'GCN' or 'GraphConv', specifying which type of GNN to use.
            d_model: int, dimension of node features.
            num_gnn_layers: int, number of GNN layers.
            dropout: float, dropout probability.
        """
        super(GNN, self).__init__()
        self.gnn_type = gnn_type
        self.d_model = d_model
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout

        # Select the appropriate convolutional layer type based on gnn_type
        if gnn_type == "GCN":
            self.conv_layers = nn.ModuleList([DenseGCNConv(d_model, d_model) for _ in range(num_gnn_layers)])
        elif gnn_type == "GraphConv":
            self.conv_layers = nn.ModuleList([DenseGraphConv(d_model, d_model) for _ in range(num_gnn_layers)])
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(num_gnn_layers)])
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj, mask):
        """
        Args:
            x: Tensor of shape (batch_size, num_nodes, d_model), node features.
            adj: Tensor of shape (batch_size, num_nodes, num_nodes), adjacency matrix.
            mask: Tensor of shape (batch_size, num_nodes, num_nodes), adjacency mask.
            add_loop: bool, whether to add self-loops to the adjacency matrix.

        Returns:
            Tensor of shape (batch_size, d_model), graph-level representation.
        """
        for i, conv in enumerate(self.conv_layers):
            # Apply the convolution layer
            out = conv(x=x, adj=adj, mask=mask)

            # Residual connection
            out = out + x

            # Batch normalization
            out = self.batch_norms[i](out.transpose(1, 2)).transpose(1, 2)  # (B, N, d_model) -> (B, d_model, N) -> (B, N, d_model)

            # ReLU activation
            out = self.relu(out)

            # Dropout
            out = self.dropout_layer(out)

            # Update x for the next layer
            x = out  # (batch_size, num_nodes, d_model)

        # Perform global mean pooling to obtain graph-level features
        readout = torch.mean(out, dim=1)  # Average pooling across nodes
        return readout
    
    
class BaselineModel(nn.Module):
    def __init__(self, embedder, encoder, classifier, max_num_encs):
        super(BaselineModel, self).__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.classifier = classifier
        self.max_num_encs = max_num_encs
        
    def embed(self, x):
        return self.embedder(x)
    
    def encode(self, x, pad_masks, causal_masks = None, priors = None):
        # the input is determined by the type of encoder
        # if time series model, take the average of the embedding of the encounter 
        if isinstance(self.encoder, RNNLayer):
            x = x.view(x.size(0), self.max_num_encs, -1, x.size(2))  
            x = torch.mean(x, dim=2)  # shape [batch_size, max_num_encs * 2, d_model]
            output = self.encoder(x, mask = None)
        elif isinstance(self.encoder, GNN):
            output = self.encoder(x = x, adj = priors, mask = pad_masks)
        elif isinstance(self.encoder, Transformer.Transformer):
            output = self.encoder(seq_emds = x, pad_masks = pad_masks, causal_masks = causal_masks)
        else:
            output = self.encoder(x, pad_masks)

        # the output is determined by the type of encoder
        if isinstance(self.encoder, RNNLayer):
            output = output[1]
        else:
            pass

        return output
    
    def classify(self, x):
        return self.classifier(x)
    
    def forward(self, x, pad_masks, causal_masks = None, priors = None):
        x = self.embed(x = x) 
        x = self.encode(x = x, pad_masks = pad_masks, causal_masks = causal_masks, priors = priors) 
        logits = self.classify(x)
        return logits
    
    
class BaselineGCT(nn.Module):
    def __init__(self, embedder, encoder, classifier):
        super(BaselineGCT, self).__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.classifier = classifier
        
    def embed(self, x):
        return self.embedder(x)
    
    def encode(self, x, pad_masks, causal_masks, priors):
        output = self.encoder(x, pad_masks, causal_masks, priors)
        return output
    def classify(self, x):
        return self.classifier(x)
    
    def forward(self, x, pad_masks, causal_masks, priors):
        x = self.embed(x = x) 
        x, KLD_loss = self.encode(x = x, pad_masks = pad_masks, causal_masks = causal_masks, priors = priors) 
        logits = self.classify(x)
        return logits, KLD_loss

    
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_classifier_layers=1):
        """
        Args:
            input_dim (int): The number of input features.
            num_classes (int): The number of output classes.
        """
        super(LinearClassifier, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_classifier_layers - 1)] 
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


def CV_results_statistics(fold_test_results: dict, confidence_level=0.95):
    """
    Organize a list of metric dictionaries into a dict with mean and confidence interval ranges.
    
    Args:
        results_list (list of dict): List of dictionaries with performance metrics.
        confidence_level (float): Confidence level for the confidence interval (default 0.95).
    
    Returns:
        dict: Organized dictionary with metrics, mean, and confidence intervals as ranges.
    """
    # for each model, we derive the statistics of the performance metrics across the folds
    model_performance = {}
    for (name, results_list) in fold_test_results.items():
        # Initialize the results dictionary
        aggregated_results = {}
        
        # Extract all metrics
        metrics = results_list[0].keys()
        
        # Calculate mean and CI for each metric
        for metric in metrics:
            # Extract values for the current metric
            values = np.array([res[metric] for res in results_list])
            
            # Calculate mean
            mean = np.mean(values)
            
            # Calculate confidence interval range
            n = len(values)
            stderr = sem(values)
            t_critical = t.ppf((1 + confidence_level) / 2, df=n-1)  # Two-tailed t critical value
            margin = t_critical * stderr
            lower_bound = mean - margin
            upper_bound = mean + margin
            
            # Store in the dictionary
            aggregated_results[metric] = {
                'mean': f"{mean:.4f}",
                '95% CI': f"{lower_bound:.4f} - {upper_bound:.4f}"
            }
        model_performance[name] = aggregated_results
    return model_performance


def cross_validation(dataset: TensorDataset, n_splits: int, hp_dict: dict, device):
    # Set up k-fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=888)
    ys = np.array([pat[-1].item() for pat in dataset]) # get the labels for stratifiedKFold
    fold_test_results = {}
    
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset, ys)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Split train+val indices into train and validation sets
        train_size = int(0.8 * len(train_val_idx))
        val_size = len(train_val_idx) - train_size
        train_idx, val_idx = torch.utils.data.random_split(train_val_idx, 
                                                           [train_size, val_size], 
                                                           torch.Generator().manual_seed(42))
        
        # Create DataLoaders
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=hp_dict['batch_size'], shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=hp_dict['batch_size'], shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=hp_dict['batch_size'], shuffle=False)
        
        model_dict = make_all_models(hp_dict, device)
        
        optimizer_dict = {}
        scheduler_dict = {}
        for name, model in model_dict.items():
            optimizer_dict[name] = torch.optim.Adam(model.parameters(), lr=hp_dict['lr'], weight_decay=hp_dict['weight_decay'])
            scheduler_dict[name] = optim.lr_scheduler.StepLR(optimizer_dict[name], step_size=hp_dict['scheduler_step'], gamma=hp_dict['scheduler_rate'])
            
        # best model during the training in this fold
        best_model_fold_dict = train(model_dict, train_loader, val_loader, optimizer_dict, 
                                     scheduler_dict, hp_dict, device = device)
        
        for name, best_model_fold in best_model_fold_dict.items():
            test_result_dict = test(best_model_fold, test_loader, device, 'test')
            if name not in fold_test_results.keys():
                fold_test_results[name] = []
            fold_test_results[name].append(test_result_dict)
            
    return CV_results_statistics(fold_test_results)



def train(model_dict, train_loader, val_loader, 
          optimizer_dict, scheduler_dict, hp_dict, device):
    
    best_val_AUPRC_dict = {name: 0 for name in model_dict.keys()}
    best_model_dict = {name: None for name in model_dict.keys()}
    best_model_epoch_dict = {name: 0 for name in model_dict.keys()}   
    
    # train the model
    num_epochs = hp_dict['epochs']
    for epoch in range(num_epochs):
        print("=====================================================================")
        print(f"\tEpoch {epoch + 1}/{num_epochs}")
        for name, model in model_dict.items():
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                # get the batch data and move it to the device
                batch = [item.to(device) for item in batch]
                _, all_code_ints_batch, all_pad_masks_batch, all_causal_masks_batch, \
                    all_intervals_batch, prior_matrices_batch, all_labels_batch = batch
                
                # zero the parameter gradients for each model's optimizer
                optimizer_dict[name].zero_grad()
                
                # forward pass
                if isinstance(model, BaselineModel):
                    logits = model(x = all_code_ints_batch, pad_masks = all_pad_masks_batch, 
                                   causal_masks = all_causal_masks_batch, priors = prior_matrices_batch)
                    # compute the loss
                    loss = F.nll_loss(logits, all_labels_batch)
                
                elif isinstance(model, BaselineGCT):
                    logits, KLD_loss = model(x = all_code_ints_batch, pad_masks = all_pad_masks_batch, 
                                             causal_masks = all_causal_masks_batch, priors = prior_matrices_batch)
                    # compute the loss
                    cls_loss = F.nll_loss(logits, all_labels_batch)
                    loss = cls_loss + hp_dict['GCT_KLD_coef'] * KLD_loss
                    
                # if the model is an ablation model, we need to handle the loss differently
                elif isinstance(model, DeePJ.DeepJourney) or (isinstance(model, DeePJ_ablation.DeepJourneyAblation) and model.use_TE == False):
                    logits, _, _, _, KLD_loss, link_loss, ent_loss = model(code_ints = all_code_ints_batch, 
                                                                           pad_masks = all_pad_masks_batch, 
                                                                           causal_masks = all_causal_masks_batch, 
                                                                           enc_intervals = all_intervals_batch, 
                                                                           prior_matrices = prior_matrices_batch)
                    # compute the loss
                    cls_loss = F.nll_loss(logits, all_labels_batch)
                    weighted_KLD_loss = hp_dict['deepj_KLD_coef'] * KLD_loss
                    weighted_link_loss = hp_dict['deepj_link_coef'] * link_loss
                    weighted_ent_loss = hp_dict['deepj_ent_coef'] * ent_loss
                    loss = cls_loss + weighted_KLD_loss + weighted_link_loss + weighted_ent_loss
                
                
                elif isinstance(model, DeePJ_ablation.DeepJourneyAblation) and model.use_SL == False:
                    logits, _, _, _, link_loss, ent_loss = model(code_ints = all_code_ints_batch,
                                                                 pad_masks = all_pad_masks_batch,
                                                                 causal_masks = all_causal_masks_batch, 
                                                                 enc_intervals = all_intervals_batch,
                                                                 prior_matrices = prior_matrices_batch)
                    # compute the loss
                    cls_loss = F.nll_loss(logits, all_labels_batch)
                    weighted_link_loss = hp_dict['deepj_link_coef'] * link_loss
                    weighted_ent_loss = hp_dict['deepj_ent_coef'] * ent_loss
                    loss = cls_loss + weighted_link_loss + weighted_ent_loss
                
                elif isinstance(model, DeePJ_ablation.DeepJourneyAblation) and model.use_GP == False:
                    logits, _, KLD_loss = model(code_ints = all_code_ints_batch,
                                                pad_masks = all_pad_masks_batch,
                                                causal_masks = all_causal_masks_batch, 
                                                enc_intervals = all_intervals_batch,
                                                prior_matrices = prior_matrices_batch)
                    # compute the loss
                    cls_loss = F.nll_loss(logits, all_labels_batch)
                    weighted_KLD_loss = hp_dict['deepj_KLD_coef'] * KLD_loss
                    loss = cls_loss + weighted_KLD_loss
            
                # all the loss conditions are covered, run the backward pass
                loss.backward()
                optimizer_dict[name].step()
                train_loss += loss.item()
            
            scheduler_dict[name].step()
            current_lr = optimizer_dict[name].param_groups[0]['lr']
            print(f"\t{name}, Loss: {train_loss / len(train_loader):.4f}, LR: {current_lr:.6f}")
        
            # test the model on the validation set, if the validation AUPRC is improved, save the model
            _ = test(model, train_loader, device, 'train')
            
            if val_loader is not None:
                val_result = test(model, val_loader, device, 'val')
                if val_result['AUPRC'] > best_val_AUPRC_dict[name]:
                    best_val_AUPRC_dict[name] = val_result['AUPRC']
                    best_model_dict[name] = c(model)
                    best_model_epoch_dict[name] = epoch + 1
                  
    print(f'\tBest model saved at epoch:', best_model_epoch_dict)
    return best_model_dict


def test(model, dataloader, device, mode = 'val'):
    model.eval()
    all_labels = []
    all_preds = []
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = [item.to(device) for item in batch]
            _, all_code_ints_batch, all_pad_masks_batch, all_causal_masks_batch, \
                all_intervals_batch, prior_matrices_batch, all_labels_batch = batch
            
            # Forward pass
            if isinstance(model, BaselineModel):
                logits = model(x = all_code_ints_batch, pad_masks = all_pad_masks_batch, 
                               causal_masks = all_causal_masks_batch, priors = prior_matrices_batch)
            elif isinstance(model, BaselineGCT):
                logits, _ = model(x = all_code_ints_batch, pad_masks = all_pad_masks_batch, 
                                  causal_masks = all_causal_masks_batch, priors = prior_matrices_batch)
            elif isinstance(model, DeePJ.DeepJourney) or isinstance(model, DeePJ_ablation.DeepJourneyAblation):
                outputs = model(code_ints = all_code_ints_batch, 
                                pad_masks = all_pad_masks_batch, 
                                causal_masks = all_causal_masks_batch, 
                                enc_intervals = all_intervals_batch, 
                                prior_matrices = prior_matrices_batch)
                logits = outputs[0]
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            # Store results
            all_labels.extend(all_labels_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())  # Logits are log softmaxed
    
    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)
    
    # Metrics
    accuracy = np.mean(all_labels == all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Compute AUROC
    if len(np.unique(all_labels)) > 2:
        # Multi-class case
        probs = np.exp(all_logits)  # Convert log softmaxed logits to probabilities
        AUROC = roc_auc_score(all_labels, probs, multi_class='ovr')
        AUPRC = average_precision_score(all_labels, probs, average='macro')
    else:
        # Binary case
        probs = np.exp(all_logits)[:, 1]  # Probability for the positive class
        AUROC = roc_auc_score(all_labels, probs)
        AUPRC = average_precision_score(all_labels, probs)
    
    # Print results
    print(f"\t\tOn {mode} set - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUROC: {AUROC:.4f}, AUPRC: {AUPRC:.4f}")
    
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'AUROC': AUROC, 'AUPRC': AUPRC}


def make_all_models(hp_dict, max_num_encs, device):
    embedder = Embedder(d_model = hp_dict['d_model'], vocab_size = hp_dict['vocab_size'])
    classifier = LinearClassifier(input_dim = hp_dict['d_model'], num_classes = hp_dict['num_classes'], 
                                  num_classifier_layers = hp_dict['num_classifier_layers'])
    gru_layers = RNNLayer(input_size = hp_dict['d_model'], hidden_size = hp_dict['d_model'], rnn_type = 'GRU', 
                          num_layers = hp_dict['num_encoder_layers'], dropout = hp_dict['dropout'], bidirectional = False)
    lstm_layers = RNNLayer(input_size = hp_dict['d_model'], hidden_size = hp_dict['d_model'], rnn_type = 'LSTM', 
                          num_layers = hp_dict['num_encoder_layers'], dropout = hp_dict['dropout'], bidirectional = False)
    tf_layers = Transformer.Transformer(d_model = hp_dict['d_model'], dropout = hp_dict['dropout'], num_layers = hp_dict['num_encoder_layers'])
    deepr_layers = DeeprLayer(feature_size = hp_dict['d_model'], hidden_size = hp_dict['d_model'])
    gcn_layers = GNN(gnn_type = 'GCN', d_model = hp_dict['d_model'], num_gnn_layers = hp_dict['num_gnn_layers'], 
                     dropout = hp_dict['dropout'])
    graphconv_layers = GNN(gnn_type = 'GraphConv', d_model = hp_dict['d_model'], num_gnn_layers = hp_dict['num_gnn_layers'], 
                           dropout = hp_dict['dropout'])
    gct_layers = GCT.GCT(d_model = hp_dict['d_model'], dropout = hp_dict['dropout'], num_layers = hp_dict['num_encoder_layers'])


    deepr = BaselineModel(c(embedder), c(deepr_layers), c(classifier), max_num_encs).to(device)
    gru = BaselineModel(c(embedder), c(gru_layers), c(classifier), max_num_encs).to(device)
    lstm = BaselineModel(c(embedder), c(lstm_layers), c(classifier), max_num_encs).to(device)
    tf = BaselineModel(c(embedder), c(tf_layers), c(classifier), max_num_encs).to(device)
    gcn = BaselineModel(c(embedder), c(gcn_layers), c(classifier), max_num_encs).to(device)
    graph_conv = BaselineModel(c(embedder), c(graphconv_layers), c(classifier), max_num_encs).to(device)

    gct = BaselineGCT(c(embedder), c(gct_layers), c(classifier)).to(device)
    
    deepj = DeePJ.make_deepj_model(hp_dict['d_model'], hp_dict['num_encoder_layers'], hp_dict['num_classifier_layers'], 
                                   hp_dict['dropout'], hp_dict['vocab_size'], hp_dict['max_elapsed_time'], hp_dict['num_deepj_graph_clusters'],
                                   hp_dict['num_classes']).to(device)
    
    deepj_wo_TE = DeePJ_ablation.make_deepj_ablation_model(hp_dict['d_model'], hp_dict['num_encoder_layers'], hp_dict['num_classifier_layers'], 
                                   hp_dict['dropout'], hp_dict['vocab_size'], hp_dict['max_elapsed_time'], hp_dict['num_deepj_graph_clusters'],
                                   hp_dict['num_classes'], use_TE = False, use_SL = True, use_GP = True).to(device)
    deepj_wo_SL = DeePJ_ablation.make_deepj_ablation_model(hp_dict['d_model'], hp_dict['num_encoder_layers'], hp_dict['num_classifier_layers'], 
                                   hp_dict['dropout'], hp_dict['vocab_size'], hp_dict['max_elapsed_time'], hp_dict['num_deepj_graph_clusters'],
                                   hp_dict['num_classes'], use_TE = True, use_SL = False, use_GP = True).to(device)
    deepj_wo_GP = DeePJ_ablation.make_deepj_ablation_model(hp_dict['d_model'], hp_dict['num_encoder_layers'], hp_dict['num_classifier_layers'], 
                                   hp_dict['dropout'], hp_dict['vocab_size'], hp_dict['max_elapsed_time'], hp_dict['num_deepj_graph_clusters'],
                                   hp_dict['num_classes'], use_TE = True, use_SL = True, use_GP = False).to(device)

    return {'GraphConv': graph_conv, 'GCN': gcn, 'GRU': gru, 'LSTM': lstm, 'Deepr': deepr, 'Transformer': tf, 'GCT': gct, 
            'DeePJ/TE': deepj_wo_TE, 'DeePJ/SL': deepj_wo_SL, 'DeePJ/GP': deepj_wo_GP, 'DeePJ': deepj}
    
    
