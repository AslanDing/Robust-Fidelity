import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_add_pool

class GCN(torch.nn.Module):
    def __init__(self, inpu_feature, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(inpu_feature, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_mask=None):  # Add edge_mask as an argument
        x = self.conv1(x, edge_index, edge_weight=edge_mask) # Apply mask in conv layer
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_mask) # Apply mask in conv layer
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = global_add_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



# use a graph as an example
edge_index = torch.randint(0, 25, (2, 50))
batch = torch.zeros(25).to(torch.long)
x = torch.randn(25, 1)
y = torch.randint(0, 1, (1,))
data = Data(x=x, edge_index=edge_index, y=y)

# define model 
model = GCN(1,20,2)
ori_pred = model(x,edge_index,batch)[0,y].item()
print(ori_pred)

# generate a sample of explanation
edge_mask = np.random.randn(50, 1)


# define top_k 
top_k_ratio = 0.1

top_k = int(edge_mask.shape[0] * top_k_ratio)
idxes = np.argsort(edge_mask, axis=0)


# fidelity+  alpha_1 = 0.1
alpha_1 = 0.1
sample_number = 50
bin_mask = np.ones_like(edge_mask)
bin_mask[idxes[top_k:]] = alpha_1

# union_masks = torch.rand([sample_number, edge_mask.shape[0]])
# union_masks = torch.where(union_masks > alpha_1, torch.ones_like(union_masks),
#                             torch.zeros_like(union_masks))
# union_masks = union_masks*(1-bin_mask) + bin_mask
fid_plus_list = []
for i in range(sample_number):
    mask_sample = torch.bernoulli(torch.from_numpy(bin_mask).float())  # sampling 
    fid_tmp =  model(x,edge_index,batch,edge_mask=mask_sample)[0,y].item()
    fid_plus_list.append(ori_pred-fid_tmp)
fid_plus = np.mean(fid_plus_list)

print(fid_plus)


# fidelity-  alpha_2 = 0.1
alpha_2 = 0.1
sample_number = 50
bin_mask = np.ones_like(edge_mask) * alpha_2
bin_mask[idxes[top_k:]] = 1

fid_minus_list = []
for i in range(sample_number):
    mask_sample = torch.bernoulli(torch.from_numpy(bin_mask).float())
    fid_tmp =  model(x,edge_index,batch,edge_mask=mask_sample)[0,y].item()
    fid_minus_list.append(ori_pred-fid_tmp)
fid_minus = np.mean(fid_minus_list)

print(fid_minus)


