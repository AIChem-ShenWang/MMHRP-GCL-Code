import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class GATU(nn.Module):
    def __init__(self, node_feature_num, channels, heads):
        super(GATU, self).__init__()
        self.conv1 = pyg_nn.GATConv(node_feature_num, channels[0], heads=heads)
        self.norm1 = nn.BatchNorm1d(channels[0] * heads)
        self.conv2 = pyg_nn.GATConv(channels[0] * heads, channels[1], heads=heads)

    def forward(self, data, 
        mask_idx : int = None,
        mask_feature: dict = None # {"feat_id" : int, "len" : int}
        ):
        x = data["x"].clone().detach().float()
        edge_index = data["edge_index"].clone().detach()
        batch = data["batch"].clone().detach()
        mask = 0

        # if mask_idx is not None:
        #     x[mask_idx, :] = x[mask_idx, :] * mask

        # GAT
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)

        # Mask Node
        if mask_idx is not None:
          if mask_feature is not None:
            feat_id = mask_feature["feat_id"]
            l = mask_feature["len"]
            x1[mask_idx, feat_id:feat_id+l] = x1[mask_idx, feat_id:feat_id+l] * mask
            x2[mask_idx, feat_id:feat_id+l] = x2[mask_idx, feat_id:feat_id+l] * mask
          else:
            x1[mask_idx, :] = x1[mask_idx, :] * mask
            x2[mask_idx, :] = x2[mask_idx, :] * mask

        x = torch.cat([x1, x2], 1)

        # Pooling
        x_mean = pyg_nn.global_mean_pool(x, batch=batch)
        x_max = pyg_nn.global_max_pool(x, batch=batch)
        x = torch.cat([x_mean, x_max], 1)

        return x

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # seq_len & batch_size order reverse
            bidirectional=True
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.device = device

    def init_hidden(self, x):
      return torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, _ = self.gru(x, h0) # x is input, size (batch, seq_len, input_size)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class MMHRP_GCL(nn.Module):
    def __init__(self,
                 GraphEncoder: dict = None,
                 TextEncoder: dict = None,
                 ModalityAlignment: dict = None,
                 Decoder: dict = None,
                 emb_size: int = 128,
                 device: str = torch.device('cuda')
                 ):
        super(MMHRP_GCL, self).__init__()

        # device
        self.device = device
        self.GraphEncoder = GraphEncoder
        self.TextEncoder = TextEncoder

        # 1.Encoder Parser
        if GraphEncoder is None and TextEncoder is None:
            raise Exception("No Encoder")

        # 1.1 GraphEncoder Parser
        if GraphEncoder is not None:
          Graph_params = ["NodeFeatNum", "Channels", "Heads"]
          for key in GraphEncoder.keys():
            if key not in Graph_params:
                raise Exception("%s is not the param in Graph Encoder")

            if key == "NodeFeatNum":
                NodeFeatNum =  GraphEncoder[key] # 8
            if key == "Channels":
                GAT_Channels = GraphEncoder[key] # [32, 64]
                assert len(GAT_Channels) == 2
            if key == "Heads":
                GAT_Heads = GraphEncoder[key] # 4

          # GATU output size
          GATU_OutSize = 0
          for i in GAT_Channels:
            GATU_OutSize += i * GAT_Heads
          self.GATU_OutSize = GATU_OutSize * 2

          # GraphEncoder
          self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
          self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
          # GATU for Reactants and Products
          self.ReaProEncoder = nn.Sequential(
            self.GATU_ReaPro,
            nn.Linear(self.GATU_OutSize, emb_size)
          )
          # GATU for Catalysts and Solvents
          self.CatSolEncoder = nn.Sequential(
            self.GATU_CatSol,
            nn.Linear(self.GATU_OutSize, emb_size)
          )
        # 1.2 TextEncoder Parser

        if TextEncoder is not None:
          Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
          for key in TextEncoder.keys():
            if key not in Text_params:
                raise Exception("%s is not the param in Text Encoder")

            if key == "SmiFeatNum":
                self.SmiFeatNum = TextEncoder[key] # 128
            if key == "Heads":
                Trans_Heads = TextEncoder[key] # 4

            if key == "BigruChannels":
                BigruChannels =  TextEncoder[key] # [128, 128]
                assert len(BigruChannels) == 2
                bigru_input, bigru_hidden = BigruChannels

            if key == "BigruNumlayer":
                bigru_num_layers = TextEncoder[key] # 2

          # TextEncoder
          self.trans = nn.TransformerEncoderLayer(d_model=self.SmiFeatNum, nhead=Trans_Heads, batch_first=True, norm_first=True)
          self.bigru = BiGRU(bigru_input, bigru_hidden, bigru_num_layers, emb_size, device)
          self.RxnSmiEncoder = nn.Sequential(
            self.trans,
            self.bigru,
            nn.ReLU()
          )

        # total_emb_size
        total_emb_size = 0
        if GraphEncoder is not None:
            total_emb_size += emb_size * 2
        if TextEncoder is not None:
            total_emb_size += emb_size

        # 2.Modality Alignment Parser
        if ModalityAlignment is not None:
            MA_params = ["Heads"]
            for key in ModalityAlignment.keys():
                if key not in MA_params:
                    raise Exception("%s is not the param in Modality Alignment")

                if key == "Heads":
                    MA_Heads = ModalityAlignment[key]  # 4

            # Modality Alignment
            self.ModalityAlignment = ModalityAlignment
            if self.ModalityAlignment:
                self.MA = nn.TransformerEncoderLayer(d_model=total_emb_size, nhead=MA_Heads)

        # 3.Decoder Parser
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception("%s is not the param in Decoder")

            if key == "Channels":
                Decoder_Channels = Decoder[key] # [1000, 500, 100]
                assert len(Decoder_Channels) == 3

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_emb_size, Decoder_Channels[0]),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    def forward(self, x):
        # MMHRP
        if self.GraphEncoder is not None and self.TextEncoder is not None:
          # import data
          ReaPro_data, CatSol_data, RxnSmi = x
          # Graph Modality
          graph_emb = torch.cat([self.ReaProEncoder(ReaPro_data),
                               self.CatSolEncoder(CatSol_data)], dim=1)
          # Text Modality
          text_embed = self.RxnSmiEncoder(RxnSmi)
          x = torch.cat([graph_emb, text_embed], dim=1)

        # No Graph Modality
        if self.GraphEncoder is None:
          RxnSmi = x
          x = self.RxnSmiEncoder(RxnSmi)
        
        # No Text Modality
        if self.TextEncoder is None:
          ReaPro_data, CatSol_data = x
          x = torch.cat([self.ReaProEncoder(ReaPro_data),
                               self.CatSolEncoder(CatSol_data)], dim=1)

        # Modality Alignment
        if self.ModalityAlignment is not None:
            x = self.MA(x)

        # Decoder
        x = self.decoder(x)

        return x

class MMHRP_GCL_Explanation(nn.Module):
    def __init__(self,
                 GraphEncoder: dict = None,
                 TextEncoder: dict = None,
                 ModalityAlignment: dict = None,
                 Decoder:dict = None,
                 emb_size: int = 128,
                 device: str = torch.device('cuda')
                 ):
        super(MMHRP_GCL_Explanation, self).__init__()

        # device
        self.device = device

        # 1. Encoder Parser
        if GraphEncoder is None and TextEncoder is None:
            raise Exception("No Encoder")

        # 1.1 GraphEncoder Parser
        Graph_params = ["NodeFeatNum", "Channels", "Heads"]
        for key in GraphEncoder.keys():
            if key not in Graph_params:
                raise Exception("%s is not the param in Graph Encoder")

            if key == "NodeFeatNum":
                NodeFeatNum =  GraphEncoder[key] # 8
            if key == "Channels":
                GAT_Channels = GraphEncoder[key] # [32, 64]
                assert len(GAT_Channels) == 2
            if key == "Heads":
                GAT_Heads = GraphEncoder[key] # 4

        # GATU output size
        GATU_OutSize = 0
        for i in GAT_Channels:
            GATU_OutSize += i * GAT_Heads
        self.GATU_OutSize = GATU_OutSize * 2

        # GraphEncoder
        # GATU for Reactants and Products
        self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        self.ReaProLinear = nn.Linear(self.GATU_OutSize, emb_size)
        self.CatSolLinear = nn.Linear(self.GATU_OutSize, emb_size)

        # TextEncoder Parser
        Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
        for key in TextEncoder.keys():
            if key not in Text_params:
                raise Exception("%s is not the param in Text Encoder")

            if key == "SmiFeatNum":
                self.SmiFeatNum =  TextEncoder[key] # 128
            if key == "Heads":
                Trans_Heads = TextEncoder[key] # 4

            if key == "BigruChannels":
                BigruChannels =  TextEncoder[key] # [128, 128]
                assert len(BigruChannels) == 2
                bigru_input, bigru_hidden = BigruChannels

            if key == "BigruNumlayer":
                bigru_num_layers =  TextEncoder[key] # 1

        # 1.2 TextEncoder
        self.trans = nn.TransformerEncoderLayer(d_model=self.SmiFeatNum, nhead=Trans_Heads, batch_first=True, norm_first=True)
        self.bigru = BiGRU(bigru_input, bigru_hidden, bigru_num_layers, emb_size, device)
        self.RxnSmiEncoder = nn.Sequential(
          self.trans,
          self.bigru,
          nn.ReLU()
        )

        # total_emb_size
        total_emb_size = 0
        if GraphEncoder is not None:
            total_emb_size += emb_size * 2
        if TextEncoder is not None:
            total_emb_size += emb_size

        # 2.Modality Alignment Parser
        MA_params = ["Heads"]
        for key in ModalityAlignment.keys():
            if key not in MA_params:
                raise Exception("%s is not the param in Modality Alignment")

            if key == "Heads":
                MA_Heads = ModalityAlignment[key]  # 4
        self.MA = nn.TransformerEncoderLayer(d_model=emb_size, nhead=MA_Heads)

        # 3.Decoder Parser
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception("%s is not the param in Decoder")

            if key == "Channels":
                Decoder_Channels = Decoder[key]  # [1000, 500, 100]
                assert len(Decoder_Channels) == 3

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_emb_size, Decoder_Channels[0]),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    def forward(self, ReaPro_x,
                ReaPro_edge_index,
                ReaPro_batch,
                CatSol_x,
                CatSol_edge_index,
                CatSol_batch,
                RxnSmi,
                ReaPro_MaskNodeIdx=None,
                CatSol_MaskNodeIdx=None):

        # import data
        ReaPro_data = Data(x=ReaPro_x, edge_index=ReaPro_edge_index, batch=ReaPro_batch)
        CatSol_data = Data(x=CatSol_x, edge_index=CatSol_edge_index, batch=CatSol_batch)

        # Graph Modality
        graph_emb = torch.cat(
          [self.ReaProLinear(self.GATU_ReaPro(ReaPro_data, ReaPro_MaskNodeIdx)),
            self.CatSolLinear(self.GATU_CatSol(CatSol_data, CatSol_MaskNodeIdx))],
            dim=1)

        # Text Modality
        text_embed = self.RxnSmiEncoder(RxnSmi)

        # Modality Alignment
        x = torch.cat([graph_emb, text_embed], dim=1)
        x = self.MA(x)

        # Decoder
        x = self.decoder(x)

        return x

# Model Evaluation Function
import numpy as np
from sklearn.metrics import mean_absolute_error

def RMSE(pred, true):
    diff_2 = (pred - true)**2
    return np.sqrt(diff_2.mean())

def R2(pred, true):
    u = ((true - pred) ** 2).sum()
    v = ((true - true.mean()) ** 2).sum()
    r2 = 1 - u / v
    return r2

def MAE(pred, true):
    return mean_absolute_error(true, pred)