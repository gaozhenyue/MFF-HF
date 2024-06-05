import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

import math
from torch.nn import CrossEntropyLoss, MSELoss

import torchvision.models as models

class MLP2(nn.Module):
    """mlp can specify number of hidden layers and hidden layer channels"""

    def __init__(
        self,
        input_dim,
        output_dim,
        act="relu",
        num_hidden_lyr=2,
        dropout_prob=0.5,
        return_layer_outs=False,
        hidden_channels=None,
        bn=False,
    ):
        super().__init__()
        self.out_dim = output_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.return_layer_outs = return_layer_outs
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels"
            )
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.act_name = act
        self.activation = create_act(act)
        self.layers = nn.ModuleList(
            list(
                map(
                    self.weight_init,
                    [
                        nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                        for i in range(len(self.layer_channels) - 2)
                    ],
                )
            )
        )
        final_layer = nn.Linear(self.layer_channels[-2], self.layer_channels[-1])
        self.weight_init(final_layer, activation="linear")
        self.layers.append(final_layer)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList(
                [torch.nn.BatchNorm1d(dim) for dim in self.layer_channels[1:-1]]
            )

    def weight_init(self, m, activation=None):
        if activation is None:
            activation = self.act_name
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
        return m

    def forward(self, x):
        """
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        layer_inputs = [x]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(self.dropout(output))

        # model.store_layer_output(self, layer_inputs[-1])
        if self.return_layer_outs:
            return layer_inputs[-1], layer_inputs
        else:
            return layer_inputs[-1]


def calc_mlp_dims(input_dim, division=2, output_dim=1):
    dim = input_dim
    dims = []
    while dim > output_dim:
        dim = dim // division
        dims.append(int(dim))
    dims = dims[:-1]
    return dims


def create_act(act, num_parameters=None):
    if act == "relu":
        return nn.ReLU()
    elif act == "prelu":
        return nn.PReLU(num_parameters)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":

        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError("Unknown activation function {}".format(act))


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def hf_loss_func(inputs, classifier, labels, num_labels, class_weights):
    logits = classifier(inputs)
    if type(logits) is tuple:
        logits, layer_outputs = logits[0], logits[1]
    else:  # simple classifier
        layer_outputs = [inputs, logits]
    if labels is not None:
        if num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss(weight=class_weights)
            labels = labels.long()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    else:
        return None, logits, layer_outputs

    return loss, logits, layer_outputs

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), attn

# transformer

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        x = self.embeds(x)

        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = x + attn_out
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)
# mlp

class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DAG_combiner(nn.Module):
    def __init__(self, dims, beta=0.2):
        super().__init__()
        self.y_layers = []
        self.g_layers = []
        self.h_layers = []

        self.x_dims = []
        self.y_dims = []
        
        self.beta = beta

        self.drop_out = nn.Dropout(0.1)
        self.act_func = nn.ReLU()

        for i, d in enumerate(dims):
            x_dim = dims[i]
            dims_copy = dims.copy()
            dims_copy.pop(i)

            self.x_dims.append(x_dim)
            self.y_dims.append(dims_copy)

            g_layer = []
            for j in dims_copy:
                g_layer.append(nn.Linear(x_dim + min(x_dim, j), x_dim).to('cuda'))
            self.g_layers.append(g_layer)

            y_layer = []
            for k in dims_copy:
                y_layer.append(MLP2(k, x_dim, act='relu', num_hidden_lyr=0, dropout_prob=0.1, hidden_channels=[],
                                        return_layer_outs=False, bn=True).to('cuda'))
            self.y_layers.append(y_layer)

            h_layer = []
            for h in dims_copy:
                h_layer.append(nn.Linear(min(x_dim, h), x_dim, bias=False).to('cuda'))
            self.h_layers.append(h_layer)

    def forward(self, data):
        assert len(self.x_dims) == len(data), f'data length must be same, {len(self.x_dims)} != {len(data)} '

        combine_feats = []
        for i, x in enumerate(self.x_dims):
            x = data[i]
            data_copy = data.copy()
            data_copy.pop(i)

            combine_feat = x
            H = 0
            for j, y_dim in enumerate(self.y_dims[i]):
                y = data_copy[j]
                if y_dim > self.x_dims[i]:
                    y = self.y_layers[i][j](y)

                g_gate = self.drop_out(self.act_func(self.g_layers[i][j](torch.cat([x, y], dim=-1))))
                g_mult = g_gate * self.h_layers[i][j](y)
                H = H + g_mult

            norm = torch.norm(x, dim=1) / (torch.norm(H, dim=1)+ 1e-12)
            alpha = torch.clamp(norm * self.beta, min=0, max=1)
            combine_feat = combine_feat + alpha[:, None] * H
            combine_feats.append(combine_feat)

        return torch.cat(combine_feats, dim=-1)


class combiner(nn.Module):
    def __init__(self, x_dim, y_dim, act=None, beta=0.2):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        self.beta = beta

        self.y_layer = MLP2(self.y_dim, self.x_dim, act='relu', num_hidden_lyr=0, dropout_prob=0.1, hidden_channels=[],
                            return_layer_outs=False, bn=True, )

        self.g_layer = nn.Linear(self.x_dim + min(self.x_dim, self.y_dim), self.x_dim)
        self.dropout = nn.Dropout(0.1)
        self.act_func = nn.ReLU()

        self.h_layer = nn.Linear(min(self.x_dim, self.y_dim), self.x_dim, bias=False, )

    def forward(self, x, y):
        if self.y_dim > self.x_dim:
            y = self.y_layer(y)

        g_gate = self.dropout(self.act_func(self.g_layer(torch.cat([x, y], dim=-1))))

        g_mult = g_gate * self.h_layer(y)

        H = g_mult
        norm = torch.norm(x, dim=1) / (torch.norm(H, dim=1)+ 1e-12)
        alpha = torch.clamp(norm * self.beta, min=0, max=1)
        combined_feats = x + alpha[:, None] * H

        return combined_feats


# main class
class DAG_Tabular(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.,
            beta1=0.2,
            beta2=0.2
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.norm = nn.BatchNorm1d(num_continuous)

        # transformer
        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        self.combiner1 = combiner(self.num_categories * dim, self.num_continuous, beta=beta1)
        self.combiner2 = combiner(self.num_continuous, self.num_categories * dim, beta=beta2)

        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(self.num_categories * dim + self.num_continuous, dim_out)

    def forward(self, x_categ, x_cont, return_attn=False, return_encode=False):
        xs = []
        flat_categ = None

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ += self.categories_offset

            x, attns = self.transformer(x_categ, return_attn=True)

            flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            normed_cont = self.norm(x_cont)

        #  xs.append(flat_categ)
            xs.append(self.combiner1(flat_categ, normed_cont))
            xs.append(self.combiner2(normed_cont, flat_categ))
            
        else:
            xs.append(flat_categ)
        x = torch.cat(xs, dim=-1)    
        logits = self.fc(x)
        logits = self.sigmoid(logits)

        if not return_attn and not return_encode:
            return logits
        
        if not return_attn and return_encode:
            return logits, x

        return logits, attns

class DAG_Tabular2(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.norm = nn.BatchNorm1d(num_continuous)

        # transformer
        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # self.combiner1 = combiner(self.num_categories * dim, self.num_continuous)
        # self.combiner2 = combiner(self.num_continuous, self.num_categories * dim)

        self.dag_combiner = DAG_combiner([self.num_categories * dim, self.num_continuous])
        # self.dag_combiner = DAG_combiner([self.num_categories * dim, 20, self.num_continuous-20])

        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(self.num_categories * dim + self.num_continuous, dim_out)

    def forward(self, x_categ, x_cont, return_attn=False):
        xs = []
        flat_categ = None

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ += self.categories_offset

            x, attns = self.transformer(x_categ, return_attn=True)

            flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            normed_cont = self.norm(x_cont)

        x = self.dag_combiner([flat_categ, normed_cont])
        # x = self.dag_combiner([flat_categ, normed_cont[:, :20], normed_cont[:, 20:]])

        # x = torch.cat(xs, dim=-1)
        logits = self.fc(x)
        logits = self.sigmoid(logits)

        if not return_attn:
            return logits

        return logits, attns


class SelfAttention_combiner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention_combiner, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 添加一个额外的维度，将输入数据转换为三维
        Q = Q.unsqueeze(2)  # shape: (batch_size, 1, data_length, hidden_dim)
        K = K.unsqueeze(2)  # shape: (batch_size, 1, data_length, hidden_dim)
        V = V.unsqueeze(2)  # shape: (batch_size, 1, data_length, hidden_dim)

        # 计算注意力权重
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1]))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # 使用注意力权重对值矩阵进行加权求和
        output = torch.matmul(attention_weights, V).squeeze(1)  # shape: (batch_size, data_length, hidden_dim)

        return output.squeeze(2)


# main class
class TabTransformer_atten(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.norm = nn.BatchNorm1d(num_continuous)

        # transformer
        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        self.combiner = SelfAttention_combiner(self.num_categories * dim + self.num_continuous,
                                               self.num_categories * dim + self.num_continuous)

        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(self.num_categories * dim + self.num_continuous, 1)

    #         self.fc = nn.Linear(len(numerical_cols), 1)

    def forward(self, x_categ, x_cont, return_attn=False):
        xs = []
        flat_categ = None

        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ += self.categories_offset

            x, attns = self.transformer(x_categ, return_attn=True)

            flat_categ = x.flatten(1)

        assert x_cont.shape[
                   1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            normed_cont = self.norm(x_cont)

        xs.append(flat_categ)
        #         xs.append(self.combiner1(flat_categ, normed_cont))
        #         xs.append(self.combiner2(normed_cont, flat_categ))
        xs.append(normed_cont)

        x = torch.cat(xs, dim=-1)
        x = self.combiner(x)
        logits = self.fc(x)
        logits = self.sigmoid(logits)

        if not return_attn:
            return logits

        return logits, attns
    
 
# main class
class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
#             if exists(continuous_mean_std):
#                 assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
#             self.register_buffer('continuous_mean_std', continuous_mean_std)

#             self.norm = nn.LayerNorm(num_continuous)
            self.norm = nn.BatchNorm1d(num_continuous)


        # transformer

        self.transformer = Transformer(
            num_tokens = total_tokens,
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
#         all_dimensions = [input_size, *hidden_dimensions, dim_out]
        all_dimensions = [input_size, dim_out]

#         self.mlp = MLP(all_dimensions, act = mlp_act)
        self.sigmoid = nn.Sigmoid()
        
        self.fc = nn.Linear(7*dim +45, 1) 

    def forward(self, x_categ, x_cont, return_attn = False):
        xs = []

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ += self.categories_offset

            x, attns = self.transformer(x_categ, return_attn = True)

            flat_categ = x.flatten(1)
            xs.append(flat_categ)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'


        if self.num_continuous > 0:
            normed_cont = self.norm(x_cont)

        xs.append(normed_cont)
        x = torch.cat(xs, dim = -1)
        logits =self.fc(x)
        logits = self.sigmoid(logits)

        if not return_attn:
            return logits

        return logits, attns




class resnet_18(nn.Module):
    def __init__(self, num_classes=2):
        super(resnet_18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet18(x)

        return  x
# main class
class mm_tab(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super(mm_tab, self).__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            #             if exists(continuous_mean_std):
            #                 assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
            #             self.register_buffer('continuous_mean_std', continuous_mean_std)

            #             self.norm = nn.LayerNorm(num_continuous)
            self.norm = nn.BatchNorm1d(num_continuous)

        # transformer

        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        #         all_dimensions = [input_size, *hidden_dimensions, dim_out]
        all_dimensions = [input_size, dim_out]

        #         self.mlp = MLP(all_dimensions, act = mlp_act)
        self.sigmoid = nn.Sigmoid()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(self.num_ftrs, self.num_ftrs)

        self.combiner1 = combiner(self.num_categories * dim + self.num_continuous, self.num_ftrs, beta=0.2)
        self.combiner2 = combiner(self.num_ftrs, self.num_categories * dim + self.num_continuous, beta=0.2)

        self.fc = nn.Linear(7 * dim + 45 + self.num_ftrs, 1)

    def forward(self, x_categ, x_cont, x_img,   return_attn=False):
        xs = []

        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ += self.categories_offset

            x, attns = self.transformer(x_categ, return_attn=True)

            flat_categ = x.flatten(1)
            # xs.append(flat_categ)

        assert x_cont.shape[
                   1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            normed_cont = self.norm(x_cont)
        # xs.append(normed_cont)

        if x_img is not None:
            flat_img = self.resnet18(x_img)
        # xs.append(flat_img)

        xs.append(self.combiner1(torch.cat([flat_categ, normed_cont], dim=-1), flat_img))
        xs.append(self.combiner2(flat_img, torch.cat([flat_categ, normed_cont], dim=-1)))

        x = torch.cat(xs, dim=-1)
        logits = self.fc(x)
        logits = self.sigmoid(logits)

        if not return_attn:
            return logits

        return logits, attns


if __name__=='__main__':
    pass