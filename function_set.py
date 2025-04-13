import numpy as np
import hdf5storage as hs
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import NoneType
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.loader import DataLoader
import math
from typing import Any

def Batch_for_train(step, batchsize, device):

    #-------------create a batch---------------
    graph_list = []
    for batch_num in range(batchsize):
        road_dataset = r'./dataset/node_feature_' + str(step * batchsize + batch_num + 1) + '.mat'
        node_feature = hs.loadmat(road_dataset)['node_feature']
        node_feature = torch.tensor(node_feature, device=device)

        road_dataset = r'./dataset/edge_index_' + str(step * batchsize + batch_num + 1) + '.mat'
        edge_index = hs.loadmat(road_dataset)['edge_index']
        edge_index = torch.tensor(edge_index, device=device, dtype=torch.int64)

        graph_list.append(Data(x=node_feature, edge_index=edge_index).to(device))

        road_dataset = r'./dataset/edge_feature_' + str(step * batchsize + batch_num + 1) + '.mat'
        edge_feature = hs.loadmat(road_dataset)['edge_feature']
        edge_for_mp_tmp = torch.tensor(edge_feature[:, 0:2], device=device) #edge feature used in message passing
        edge_for_predict_tmp = torch.tensor(edge_feature, device=device)    #edge feature used in link prediction

        road_dataset = r'./dataset/node_rd_' + str(step * batchsize + batch_num + 1) + '.mat'
        node_rd_tmp = hs.loadmat(road_dataset)['node_rd']
        node_rd_tmp = torch.tensor(node_rd_tmp, device=device)

        road_dataset = r'./dataset/edge_label_' + str(step * batchsize + batch_num + 1) + '.mat'
        edge_label_temp = hs.loadmat(road_dataset)['edge_label']
        edge_label_temp = np.array(edge_label_temp, dtype=np.int64).reshape(-1, )

        if batch_num == 0:
            edge_for_mp = edge_for_mp_tmp.clone()
            edge_for_predict = edge_for_predict_tmp.clone()
            node_rd = node_rd_tmp.clone()
            edge_label = edge_label_temp.copy()
        else:
            edge_for_mp = torch.cat((edge_for_mp, edge_for_mp_tmp), 0)
            edge_for_predict = torch.cat((edge_for_predict, edge_for_predict_tmp), 0)
            node_rd = torch.cat((node_rd, node_rd_tmp), 2)
            edge_label = np.concatenate((edge_label, edge_label_temp))

    data_loader = DataLoader(graph_list, batch_size=batchsize)

    # -------------negative sampling---------------
    for graph in data_loader:
        # The edge indices in an undirected graph are reciprocal.
        all_edge_num = int(graph.edge_index.shape[1] / 2)
        train_index = np.arange(0, all_edge_num).copy()
        train_edge = graph.edge_index[:, list(train_index)]
        train_edge_label = edge_label[list(train_index)]

        num_edge_T_T = np.sum(train_edge_label == 2)  # the number of edge between target and target
        if num_edge_T_T == 0:                         # the number of edge between False and False
            num_edge_F_F = 2 * batchsize
        else:
            num_edge_F_F = 1 * num_edge_T_T
        num_edge_F_T = num_edge_T_T                   # the number of edge between False and target

        # the index of edge between False and False
        index_F_F = np.random.choice(np.where(train_edge_label == 0)[0], num_edge_F_F, replace=False, p=None)
        index_F_F = np.sort(index_F_F)

        # the index of edge between False and target
        index_F_T = np.random.choice(np.where(train_edge_label == 1)[0], num_edge_F_T, replace=False, p=None)
        index_F_T = np.sort(index_F_T)

        # the index of edge between target and target
        index_T_T = np.where(train_edge_label == 2)[0]

        train_edge = train_edge[:, list(index_T_T) + list(index_F_T) + list(index_F_F)]
        train_edge_back = torch.vstack((train_edge[1, :], train_edge[0, :]))
        train_edge = torch.hstack((train_edge, train_edge_back))
        train_edge = train_edge.to(device)

        train_edge_label = train_edge_label[list(index_T_T) + list(index_F_T) + list(index_F_F)]
        train_edge_label = np.concatenate((train_edge_label, train_edge_label))
        train_edge_label = torch.tensor(train_edge_label, device=device)

        train_edge_order = np.array(list(index_T_T) + list(index_F_T) + list(index_F_F))
        train_edge_order_back = train_edge_order + int(graph.edge_index.shape[1] / 2)
        train_edge_order_all = list(train_edge_order) + list(train_edge_order_back)
        edge_for_predict = edge_for_predict[train_edge_order_all, :]

    return graph, node_rd, train_edge, train_edge_label, edge_for_mp, edge_for_predict


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)
def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)
def zeros(value: Any):
    constant(value, 0.)

class STEFGAT(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        edge_hidden : int = 8,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.edge_hidden = edge_hidden
        self.edge_FC = Linear(2, heads * edge_hidden)
        self.lin_edge_node = Linear((out_channels+edge_hidden)*heads,out_channels*heads)

        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False, weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False, weight_initializer='glorot')

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, edge_feature = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa

        H, C = self.heads, self.out_channels

        # First, transform the input node features.
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, compute node-level attention coefficients, both for source and target nodes:
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # Only add self-loops for nodes that appear both as source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size, edge_feature = edge_feature)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor, edge_feature: Tensor) -> Tensor:
        n = x_j[0:edge_feature.shape[0], :]
        n = n.view(-1,self.heads*self.out_channels)
        n_lin = self.lin_edge_node(torch.cat([n,self.edge_FC(edge_feature)],1))
        n_lin = n_lin.relu()
        n_lin = n_lin.view(-1,self.heads,self.out_channels)
        out = torch.cat([n_lin,x_j[edge_feature.shape[0]:,:,]],0)
        return alpha.unsqueeze(-1) * out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MFLPN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.pretrain1 = Linear(1, 4)
        self.pretrain2 = Linear(1, 4)
        self.pretrain3 = Linear(4, 4)

        self.cnn1 = nn.Conv2d(1, 4, (3, 3),padding=(5,0),padding_mode='circular')
        self.cnn2 = nn.Conv2d(4, 4, (3, 1))
        self.cnn3 = nn.Conv2d(4, 4, (4, 1),stride=(2,1))
        self.cnn4 = nn.Conv2d(4, 4, (4, 1),stride=(2,1))
        self.cnn5 = nn.Conv2d(4, 4, (3, 1))
        self.cnn6 = nn.Conv2d(4, 1, (3, 3))

        self.mp1 = STEFGAT(16, 32, heads=1, dropout=0, edge_hidden = 8, edge_dim=2)
        self.mp2 = STEFGAT(32, 32, heads=1, dropout=0, edge_hidden = 8, edge_dim=2)
        self.mp3 = STEFGAT(32, 16, heads=1, dropout=0, edge_hidden = 8, edge_dim=2)

        self.edge_fc = Linear(3, 16)
        self.Line1 = Linear(16*3, 16)
        self.Line2 = Linear(16, 8)
        self.Line3 = Linear(8, 3)

    def encode(self, graph, edge_for_mp, node_rd):
        x1 = self.pretrain1(graph.x[:,0].view(-1,1))
        x1 = x1.relu()
        x2 = self.pretrain2(graph.x[:,1].view(-1,1))
        x2 = x2.relu()
        x3 = self.pretrain3(graph.x[:,2:])
        x3 = x3.relu()

        x4 = self.cnn1(node_rd.permute(2, 0, 1).unsqueeze(1))
        x4 = x4.relu()
        x5 = self.cnn2(x4)
        x5 = x5.relu()
        x6 = self.cnn3(x5)
        x6 = x6.relu()
        x7 = self.cnn4(x6)
        x7 = x7.relu()
        x8 = self.cnn5(x7)
        x8 = x8.relu()
        x9 = self.cnn6(x8)
        x9 = x9.relu()
        x9 = x9.squeeze()
        x_pretrain = torch.cat([x1,x2,x3,x9],1)

        x = self.mp1(x = x_pretrain, edge_index = graph.edge_index, edge_attr=edge_for_mp, edge_feature = edge_for_mp)
        x = x.relu()
        x = self.mp2(x = x, edge_index = graph.edge_index, edge_attr=edge_for_mp, edge_feature = edge_for_mp)
        x = x.relu()
        x = self.mp3(x = x, edge_index = graph.edge_index, edge_attr=edge_for_mp, edge_feature = edge_for_mp)
        return x

    def decode(self, z, edge_index, edge_for_predict):
        edge_v_tensor_predict = self.edge_fc(edge_for_predict)
        z0 = z[edge_index[0]]
        z1 = z[edge_index[1]]
        z0_z1 = torch.cat([z0,z1,edge_v_tensor_predict],1)
        logits = self.Line1(z0_z1)
        logits = logits.relu()
        logits = self.Line2(logits)
        logits = logits.relu()
        logits = self.Line3(logits)
        return logits



