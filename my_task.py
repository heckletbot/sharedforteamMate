import torch
import numpy as np
from torchdrug import core, tasks, layers, metrics
from torchdrug.core import Registry as R
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

@R.register("tasks.ProteinClassification")
class ProteinClassification(tasks.Task, core.Configurable):
    def __init__(self, model, criterion="ce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 num_class=0, graph_construction_model=None):
        super(ProteinClassification, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.graph_construction_model = graph_construction_model
        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [num_class])

    def preprocess(self, train_set, valid_set, test_set):
        train_set = train_set

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        target_index,target = self.target(batch)
        pred = self.predict(batch, all_loss, metric)
        metric.update(self.evaluate(pred, [target_index,target]))
        loss = F.cross_entropy(pred, target_index.long().squeeze(-1))
        metric["ce loss"] = loss

        all_loss += loss

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        label_index = batch["superfamily_label"].unsqueeze(1)
        # 对bceloss
        target = np.zeros((len(label_index),self.num_class))
        for i in range(len(label_index)):
            target[i,label_index[i]]=1
        # ce只需要index
        target_index = label_index
        # tensor in cuda, numpy
        return target_index,target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auprc":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = torch.tensor(score)

        return metric