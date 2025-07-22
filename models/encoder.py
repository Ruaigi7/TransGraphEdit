from typing import Tuple
import math
from models.model_utils import index_select_ND
from typing import Union
from torch import Tensor
from torch import nn, sum
from torch_geometric.data import Data, Batch
from typing import List, Union

from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
class MPNEncoder(nn.Module):
    """Class: 'MPNEncoder' is a message passing neural network for encoding molecules."""

    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size: int,
                 depth: int, dropout: float = 0.15, atom_message: bool = False):
        """
        Parameters
        ----------
        atom_fdim: Atom feature vector dimension.
        bond_fdim: Bond feature vector dimension.
        hidden_size: Hidden layers dimension
        depth: Number of message passing steps
        droupout: the droupout rate
        atom_message: 'D-MPNN' or 'MPNN', centers messages on bonds or atoms.
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.atom_message = atom_message

        # Input
        input_dim = self.atom_fdim if self.atom_message else self.bond_fdim
        self.w_i = nn.Linear(input_dim, self.hidden_size, bias=False)

        # Update message
        if self.atom_message:
            self.w_h = nn.Linear(
                self.bond_fdim + self.hidden_size, self.hidden_size)

        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Output
        self.W_o = nn.Sequential(
            nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), nn.ReLU())

        # # 添加残差连接相关层
        # self.use_residual = True
        # if self.use_residual:
        #     self.residual_layers = nn.ModuleList([
        #         nn.Linear(self.hidden_size, self.hidden_size)
        #         for _ in range(self.depth)
        #     ])
        #     self.layer_norms = nn.ModuleList([
        #         nn.LayerNorm(self.hidden_size)
        #         for _ in range(self.depth)
        #     ])

    def forward(self, graph_tensors: Tuple[torch.Tensor], mask: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass of the graph encoder. Encodes a batch of molecular graphs.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details, the incoming bond indices of atoms
            the index of the atom the bond is coming from, the index of the reverse bond and the undirected bond index 
            to the beginindex and endindex of the atoms.
        mask: torch.Tensor,
            Masks on nodes
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors
        # Input
        if self.atom_message:
            a2a = b2a[a2b]  # num_atoms x max_num_bonds
            f_bonds = f_bonds[:, -self.bond_fdim:]
            input = self.w_i(f_atoms)  # num_atoms x hidden
        else:
            input = self.w_i(f_bonds)  # num_bonds x hidden

        # Message passing
        # message = torch.zeros(input.size(0), self.hidden_size, device=input.device)
        message = input
        message_mask = torch.ones(message.size(0), 1, device=message.device)
        message_mask[0, 0] = 0  # first message is padding
        # # 残差连接
        # for depth in range(self.depth - 1):
        #     if self.use_residual:
        #         residual = self.residual_layers[depth](message)

        for depth in range(self.depth - 1):
            if self.atom_message:
                # num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2a)
                # num_atoms x max_num_bonds x bond_fdim
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                # num_atoms x max_num_bonds x hidden + bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                # num_atoms x hidden + bond_fdim
                message = nei_message.sum(dim=1)
                message = self.w_h(message)  # num_bonds x hidden
            else:
                # num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.gru(input, message)  # num_bonds x hidden_size
            # # 添加残差连接和层标准化
            # if self.use_residual:
            #     message = self.layer_norms[depth](message + residual)

            message = message * message_mask
            message = self.dropout_layer(message)  # num_bonds x hidden

        if self.atom_message:
            # num_atoms x max_num_bonds x hidden
            nei_a_message = index_select_ND(message, a2a)
        else:
            # num_atoms x max_num_bonds x hidden
            nei_a_message = index_select_ND(message, a2b)
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        # num_atoms x (atom_fdim + hidden)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.W_o(a_input)  # num_atoms x hidden


        if mask is None:
            mask = torch.ones(atom_hiddens.size(0), 1, device=f_atoms.device)
            mask[0, 0] = 0  # first node is padding

        return atom_hiddens * mask

class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False,
                 identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1, 2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1 + self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, num_heads=8, dropout=0.1):
        """
        Parameters:
        in_dim: input feature dimension
        hidden_dim: hidden dimension
        num_heads: number of attention heads
        dropout: dropout rate
        """
        super(MultiHeadGraphAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Define linear transformation layers
        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(in_dim, hidden_dim)
        self.v_proj = nn.Linear(in_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Add graph structure attention weights
        self.graph_weights = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.graph_weights)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Use ContraNorm instead of LayerNorm
        self.contra_norm1 = ContraNorm(
            dim=hidden_dim,
            scale=0.1,
            dual_norm=False,
            pre_norm=False,
            temp=1.0,
            learnable=False,
            positive=False,
            identity=False
        )

        self.contra_norm2 = ContraNorm(
            dim=hidden_dim,
            scale=0.1,
            dual_norm=False,
            pre_norm=False,
            temp=1.0,
            learnable=False,
            positive=False,
            identity=False
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, mask=None):
        """
        Parameters:
        x: input tensor, shape [batch_size, num_nodes, feature_dim]
        mask: (optional) mask tensor, shape [batch_size, num_nodes]
        Returns:
        output: output tensor, with the same shape as the input
        """
        batch_size, num_nodes, feature_dim = x.size()

        # 1. Compute Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Adjust dimension order
        q = q.transpose(1, 2)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # 3. Add graph structure information
        graph_attn = torch.einsum('bhnd,hd->bhn', q, self.graph_weights)
        graph_attn = graph_attn.unsqueeze(-1)
        attn_weights = attn_weights + graph_attn

        # 4. Apply mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, attn_weights.size(1), 1)
            attn_weights[~mask.bool()] = float('-inf')

        # 5. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. Compute output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_nodes, -1)

        # 7. Final linear transformation
        output = self.out_proj(attn_output)

        # 8. Use ContraNorm and residual connection
        output = self.contra_norm1(x + output)

        # 9. Feed-forward network and second ContraNorm
        output = self.contra_norm2(output + self.ffn(output))

        return output



