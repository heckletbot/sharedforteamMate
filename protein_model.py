from torchdrug import models, tasks, layers
from torchdrug.layers import geometry
import torch.nn as nn
import torch

class PDC(nn.Module):
    def __init__(self):
        super(PDC, self).__init__()
        self.gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512],
                              num_relation=7, edge_input_dim=59, num_angle_bin=8,
                              batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

    def forward(self, batch):
        graph = batch["graph"]
        output = self.gearnet_edge(graph, graph.node_feature.float())
        #pred = self.mlp(output["graph_feature"])
        return output