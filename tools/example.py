import os
import torch
import torch_geometric as ptgeom
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth
from ExplanationEvaluation.tasks.replication_table import to_torch_graph
from ExplanationEvaluation.models.model_selector_xgnn import model_selector,model_selector_extend,model_selector_contrastive

from fidelity import *

# nodes
task = 'node'
model = 'GNN'
dataset = 'syn3'
graphs, features, labels, _, _, test_mask = load_dataset(dataset)

features = torch.tensor(features).cuda()
labels = torch.tensor(labels).cuda()
graphs = to_torch_graph(graphs, task)

model, checkpoint = model_selector(model,
                                   dataset,
                                   pretrained=True,
                                   return_checkpoint=True)
explanation_labels, indices = load_dataset_ground_truth(dataset)

for index in indices:
    # find a subgraph
    explain_graph = ptgeom.utils.k_hop_subgraph(index, 3, graphs)[1]
    # explanation weight
    weight = torch.ones_like(explain_graph[0])

    # original fidelity
    fid_plus, fid_minus, fid_plus_label, fid_minus_label, sparsity = cal_fid(task,model,features,graphs,explain_graph,
                                                                             weight,
                                                                             explanation_labels[index],index)
    # new fidelity
    fid_plus_mean, fid_plus_label_mean, _ = edit_distance_gt_ratio_plus(task,model,features,graphs,
                                                            explain_graph,weight,explanation_labels[index],index,k=0.1)
    fid_minus_mean, fid_minus_label_mean, _ = edit_distance_gt_ratio_minus(task,model,features,graphs,
                                                            explain_graph,weight,explanation_labels[index],index,k=0.1)

    break


# graph
task = 'graph'
model = 'GNN'
dataset = 'ba2'
graphs, features, labels, _, _, test_mask = load_dataset(dataset)

features = torch.tensor(features).cuda()
labels = torch.tensor(labels).cuda()
graphs = to_torch_graph(graphs, task)

model, checkpoint = model_selector(model,
                                   dataset,
                                   pretrained=True,
                                   return_checkpoint=True)
explanation_labels, indices = load_dataset_ground_truth(dataset)

for index in indices:
    # find a subgraph
    explain_graph = graphs[index]
    # explanation weight
    weight = torch.ones_like(explain_graph[0])

    # original fidelity
    fid_plus, fid_minus, fid_plus_label, fid_minus_label, sparsity = cal_fid(task,model,features,graphs[index],explain_graph,
                                                                             weight,
                                                                             explanation_labels[index],index)
    # new fidelity
    fid_plus_mean, fid_plus_label_mean, _ = edit_distance_gt_ratio_plus(task,model,features,graphs,
                                                            explain_graph,weight,explanation_labels[index],index,k=0.1)
    fid_minus_mean, fid_minus_label_mean, _ = edit_distance_gt_ratio_minus(task,model,features,graphs,
                                                            explain_graph,weight,explanation_labels[index],index,k=0.1)

    break

