from torchdrug import transforms
from torchdrug import core
from torchdrug.layers import geometry
from my_model import *
from my_task import *
from prepare_data import *
# load data
truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])
dataset = ScopeSuperFamilyClassify('/home/tangwuguo/datasets/scope40', transform=transform)
#dataset = ScopeSuperFamilyClassify( '/home/tangwuguo/datasets/scope40_s', transform=transform)
train_set, valid_set, test_set = dataset.split()
print("Shape of function labels for a protein: ", dataset[0]["superfamily_label"].shape)
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))

model = ProteinClassificationNetwork(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
                         batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

task = ProteinClassification(model, graph_construction_model=graph_construction_model, num_mlp_layer=2, 
                                        num_class=2065, criterion="ce",metric=[])

optimizer = torch.optim.Adam(task.parameters(), lr=1e-2)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0],
                     batch_size=128)
solver.train(num_epoch=100)
solver.evaluate("valid")
tasks.MultipleBinaryClassification









