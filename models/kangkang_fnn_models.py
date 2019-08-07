import torch
from torch import nn

"""
### Feed Forward Network (Kang and Kang)

* **Input Data Format:** (N,D_spatial),(N,D_temporal),(N,D_environment) or (N,D) vetors are fed in independently 
* **Input Data Type:** Continuous values
Spatial feature group (35-D): demographic (9-D), housing (6-D), education (8-D), and economic (12-D).
 - we included the area of the tract - to hopefull get a density factor in as well.
  
Temporal feature group (15-D): weather (11-D), number of incidents of crime occurrence by sampling point in 2013 (1-D),
 number of incidents of crime occurrence by census tract in 2013 (1-D), number of incidents of crime occurrence by date
  in 2013 (1-D), and number of incidents of crime occurrence by census tract yesterday (1-D).

Environmental context feature group (4096-D): an image feature (4096-D).

Key extraction of the data will be done in the dataset / data loader.

* **Output Data Format:** (N,C) with C being a binary class 
* **Output Data Type:** Continuous value (number of crimes per cell)
* **Loss Function:** RMSE
"""


class KangFeedForwardNetwork(nn.Module):
    def __init__(self, spc_size=37, tmp_size=15, env_size=512, do_drop=False):
        super(KangFeedForwardNetwork, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.do_drop = do_drop

        self.spcNet = nn.Sequential(nn.Linear(spc_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU())
        self.tmpNet = nn.Sequential(nn.Linear(tmp_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU())
        self.envNet = nn.Sequential(nn.Linear(env_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU())
        self.finalNet = nn.Sequential(nn.Linear(384, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Linear(1024, 2))  # ,nn.Softmax(dim=-1))

    def forward(self, spc_vec, tmp_vec, env_vec):
        if self.do_drop:
            spc_vec = self.dropout(spc_vec)
            tmp_vec = self.dropout(tmp_vec)
            env_vec = self.dropout(env_vec)

        mid_vec = torch.cat([self.spcNet(spc_vec), self.tmpNet(tmp_vec), self.envNet(env_vec)], dim=-1)
        out_vec = self.finalNet(mid_vec)

        return out_vec
