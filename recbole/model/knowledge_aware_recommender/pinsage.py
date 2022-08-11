# @Time   : 2022/5/2
# @Author : Kuznetsova Maria
# @Email  : mr.kuznetsova98@gmail.com

r"""
PinSage
##################################################
Reference:
    Yixin Cao et al. "Unifying Knowledge Graph Learning and Recommendation:Towards a Better Understanding
    of User Preferences." in WWW 2019.

    Ying, R. et al. "Graph convolutional neural networks for web-scale recommender systems." in KDD 2018.

Reference code:
    https://arxiv.org/pdf/1806.01973 
"""

import torch
import torch.nn as nn
import torchtext
import dgl
import tqdm
import torch.nn.functional as F
import dgl.function as fn
# from torch.autograd import Variable

#from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.abstract_recommender import PinSAGERecommender
# from recbole.model.init import xavier_uniform_initialization
# from recbole.model.loss import BPRLoss, EmbMarginLoss
from recbole.utils import InputType


class PinSAGE(KnowledgeRecommender):
    r"""PinSAGE
    """

    input_type = InputType.PAIRWISE # ????



    def __init__(self, config, dataset):
        super(PinSAGE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.L1_flag = config['L1_flag']
        self.use_st_gumbel = config['use_st_gumbel']
        self.kg_weight = config['kg_weight']
        self.align_weight = config['align_weight']
        self.margin = config['margin']

        # define layers and loss
        self.proj = LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = SAGENet(hidden_dims, n_layers)
        self.scorer = ItemToItemScorer(full_graph, ntype)



# LAYERS


