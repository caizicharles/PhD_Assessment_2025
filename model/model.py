import torch
import torch.nn as nn
from pyhealth.models import RETAINLayer, StageNetLayer
from model.modules import Encoder

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'selu': nn.SELU()
}


class GRU(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.static_feats_num = configs['static_feats_num']
        self.intervention_feats_num = configs['intervention_feats_num']
        self.vital_feats_num = configs['vital_feats_num']
        self.hidden_dim = configs['hidden_dim']
        self.layer_num = configs['layer_num']
        self.out_dim = configs['out_dim']
        self.dropout = configs['dropout']

        self.static_projection = nn.Linear(self.static_feats_num, self.hidden_dim)
        self.gru = nn.GRU(input_size=self.intervention_feats_num + self.vital_feats_num,
                          hidden_size=self.hidden_dim,
                          num_layers=self.layer_num,
                          batch_first=True,
                          dropout=self.dropout)
        self.head = nn.Linear(self.hidden_dim * 2, self.out_dim)

    def forward(self, x_static, x_dynamic, fourier_coeffs):

        x_vitals = torch.fft.ifft(fourier_coeffs, dim=-1).real
        x_dynamic = torch.cat((x_dynamic, x_vitals), dim=1)

        h_static = self.static_projection(x_static)
        _, h_dynamic = self.gru(x_dynamic.transpose(-1, -2))
        h_dynamic = h_dynamic[-1]
        h_joint = torch.cat((h_static, h_dynamic), dim=-1)

        out = self.head(h_joint)

        return {'logits': out, 'reconstructed_vitals': None}


class Transformer(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.static_feats_num = configs['static_feats_num']
        self.intervention_feats_num = configs['intervention_feats_num']
        self.vital_feats_num = configs['vital_feats_num']
        self.hidden_dim = configs['hidden_dim']
        self.ff_dim = configs['ff_dim']
        self.head_num = configs['head_num']
        self.encoder_depth = configs['encoder_depth']
        self.out_dim = configs['out_dim']
        self.dropout = configs['dropout']

        self.static_projection = nn.Linear(self.static_feats_num, self.hidden_dim)
        self.dynamic_projection = nn.Linear(self.intervention_feats_num + self.vital_feats_num, self.hidden_dim)
        self.dynamic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                       nhead=self.head_num,
                                       dim_feedforward=self.ff_dim,
                                       dropout=self.dropout,
                                       batch_first=True), self.encoder_depth)

        self.head = nn.Linear(self.hidden_dim * 2, self.out_dim)

    def forward(self, x_static, x_dynamic, fourier_coeffs):

        x_vitals = torch.fft.ifft(fourier_coeffs, dim=-1).real
        x_dynamic = torch.cat((x_dynamic, x_vitals), dim=1)

        h_static = self.static_projection(x_static)
        x_dynamic = self.dynamic_projection(x_dynamic.transpose(-1, -2))
        h_dynamic = self.dynamic_encoder(x_dynamic)
        h_dynamic = h_dynamic[:, -1]
        h_joint = torch.cat((h_static, h_dynamic), dim=-1)

        out = self.head(h_joint)

        return {'logits': out, 'reconstructed_vitals': None}


class RETAIN(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.static_feats_num = configs['static_feats_num']
        self.intervention_feats_num = configs['intervention_feats_num']
        self.vital_feats_num = configs['vital_feats_num']
        self.hidden_dim = configs['hidden_dim']
        self.output_dim = configs['out_dim']
        self.dropout = configs['dropout']

        self.static_projection = nn.Linear(self.static_feats_num, self.hidden_dim)
        self.dynamic_projection = nn.Linear(self.intervention_feats_num + self.vital_feats_num, self.hidden_dim)
        self.retain = RETAINLayer(feature_size=self.hidden_dim, dropout=self.dropout)

        self.head = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x_static, x_dynamic, fourier_coeffs):

        x_vitals = torch.fft.ifft(fourier_coeffs, dim=-1).real
        x_dynamic = torch.cat((x_dynamic, x_vitals), dim=1)

        h_static = self.static_projection(x_static)
        x_dynamic = self.dynamic_projection(x_dynamic.transpose(-1, -2))
        h_dynamic = self.retain(x_dynamic)
        h_joint = torch.cat((h_static, h_dynamic), dim=-1)

        out = self.head(h_joint)

        return {'logits': out, 'reconstructed_vitals': None}


class StageNet(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.static_feats_num = configs['static_feats_num']
        self.intervention_feats_num = configs['intervention_feats_num']
        self.vital_feats_num = configs['vital_feats_num']
        self.hidden_dim = configs['hidden_dim']
        self.conv_dim = configs['hidden_dim']
        self.conv_size = configs['conv_size']
        self.levels = configs['levels']
        self.output_dim = configs['out_dim']
        self.dropout = configs['dropout']
        self.dropconnect = 0.
        self.dropres = 0.
        self.chunk_size = self.hidden_dim // self.levels
        assert self.hidden_dim % self.levels == 0

        self.static_projection = nn.Linear(self.static_feats_num, self.hidden_dim)
        self.stagenet = StageNetLayer(input_dim=self.intervention_feats_num + self.vital_feats_num,
                                      chunk_size=self.hidden_dim,
                                      conv_size=self.conv_size,
                                      levels=self.levels,
                                      dropconnect=self.dropconnect,
                                      dropout=self.dropout,
                                      dropres=self.dropres)

        self.head = nn.Linear(self.hidden_dim * (1 + self.levels), self.output_dim)

    def forward(self, x_static, x_dynamic, fourier_coeffs):

        x_vitals = torch.fft.ifft(fourier_coeffs, dim=-1).real
        x_dynamic = torch.cat((x_dynamic, x_vitals), dim=1)

        h_static = self.static_projection(x_static)

        h_dynamic, _, _ = self.stagenet(x_dynamic.transpose(-1, -2))
        h_joint = torch.cat((h_static, h_dynamic), dim=-1)

        out = self.head(h_joint)

        return {'logits': out, 'reconstructed_vitals': None}


class HIP(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.static_feats_num = configs['static_feats_num']
        self.intervention_feats_num = configs['intervention_feats_num']
        self.vital_feats_num = configs['vital_feats_num']
        self.window_size = configs['window_size']
        self.scoring_hidden_dim = configs['scoring_hidden_dim']
        self.k_coeffs = configs['k_coeffs']
        self.static_hidden_sizes = configs['static_hidden_sizes']
        self.dynamic_layers = configs['dynamic_layers']
        self.fuse_dim = configs['fuse_dim']
        self.predictor_hidden_sizes = configs['predictor_hidden_sizes']
        self.out_dim = configs['out_dim']
        self.act_fn = ACTIVATION_FUNCTIONS[configs['activation']]
        self.dropout = configs['dropout']
        self.softmax_temp = configs['softmax_temp']

        self.scoring = nn.Sequential(nn.Linear(self.window_size * 2, self.scoring_hidden_dim), self.act_fn,
                                     nn.Linear(self.scoring_hidden_dim, self.window_size))
        self.static_encoder = Encoder(input_dim=self.static_feats_num,
                                      out_dim=self.fuse_dim,
                                      hidden_sizes=self.static_hidden_sizes,
                                      act_fn=self.act_fn,
                                      dropout=self.dropout)
        self.dynamic_encoder = nn.GRU(input_size=self.intervention_feats_num + self.vital_feats_num,
                                      hidden_size=self.fuse_dim,
                                      num_layers=self.dynamic_layers,
                                      batch_first=True,
                                      dropout=self.dropout)
        self.joint_predictor = Encoder(input_dim=self.fuse_dim * 2,
                                       out_dim=self.out_dim,
                                       hidden_sizes=self.predictor_hidden_sizes,
                                       act_fn=self.act_fn)

    def STE_topk_mask(self, scores, k, temp=1.):
        soft_mask = torch.softmax(scores / temp, dim=-1)
        hard_mask = torch.zeros_like(scores)
        topk_indices = torch.topk(scores, k, dim=-1).indices
        hard_mask.scatter_(-1, topk_indices, 1)
        return hard_mask + (soft_mask - soft_mask.detach())

    def forward(self, x_static, x_dynamic, fourier_coeffs):
        '''
        x_static: (B, N_static)
        x_dynamic: (B, N_intervention, T)
        fourier_coeffs: (B, N_vitals, T)
        '''
        B = x_static.size(0)

        coeffs = torch.view_as_real(fourier_coeffs).reshape(B, self.vital_feats_num, -1)
        scores = self.scoring(coeffs)
        scores_mask = self.STE_topk_mask(scores, self.k_coeffs, temp=self.softmax_temp)
        selected_coeffs = fourier_coeffs * scores_mask

        x_vitals = torch.fft.ifft(selected_coeffs, dim=-1).real
        x_dynamic = torch.cat((x_dynamic, x_vitals), dim=1)  # (B, num_interventions+num_vitals, D)

        h_static = self.static_encoder(x_static)  # (B, D)
        _, h_dynamic = self.dynamic_encoder(x_dynamic.transpose(-1, -2))  # (B, num_interventions+num_vitals, D)
        h_dynamic = h_dynamic[-1]  # select last hidden: (B, D)
        h_joint = torch.cat((h_static, h_dynamic), dim=-1)

        out = self.joint_predictor(h_joint)

        return {'logits': out, 'reconstructed_vitals': x_vitals}


MODELS = {'HIP': HIP, 'GRU': GRU, 'Transformer': Transformer, 'RETAIN': RETAIN, 'StageNet': StageNet}
