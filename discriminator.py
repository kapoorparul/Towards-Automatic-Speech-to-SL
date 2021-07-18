
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from helpers import freeze_params
from transformer_layers import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, FusionLayer, TransformerEncoderLayer


class Classifier(nn.Module):
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size

class ClassifierLayers(Classifier):

    #pylint: disable=unused-argument
    def __init__(self,
                 src_size:int = 80,
                 trg_size:int = 150,
                 pose_time_dim:int=125,
                 aud_time_dim:int=500,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the discriminator.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(ClassifierLayers, self).__init__()
        src_size = src_size
        embedding_dim = hidden_size#emb_dim
        self.trg_embed = nn.Linear(trg_size, embedding_dim)
        

        self.src_downsample = nn.Sequential(nn.Conv1d(aud_time_dim, aud_time_dim//2, kernel_size=3, stride=2),
                                nn.Conv1d(aud_time_dim//2, aud_time_dim//2, kernel_size=3, stride=1),
                                nn.Conv1d(aud_time_dim//2, aud_time_dim//(2*2), kernel_size=3, stride=2),
                                nn.Conv1d(aud_time_dim//(2*2), aud_time_dim//(2*2), kernel_size=3, stride=1),
                                nn.Linear(16, embedding_dim))

        # build all (num_layers) layers
        self.aud_layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        self.pose_layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        self.fusion_layers = nn.ModuleList([
            FusionLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

    
        self.self_attention = TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                num_heads=num_heads, dropout=dropout)
                
    

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        
        self.feed_forward = PositionwiseFeedForward(hidden_size, ff_size=ff_size)
        size = hidden_size
        self.linear_layers = nn.Sequential(nn.Linear(size, size//2),nn.ReLU(),
                                        nn.Linear(size//2, size//4), nn.ReLU(),
                                        nn.Linear(size//4, size//8), nn.ReLU(),
                                        nn.Linear(size//8, size//16), nn.ReLU(),
                                        nn.Linear(size//16, size//32), nn.ReLU(),
                                        nn.Linear(size//32, size//64), nn.ReLU(),
                                        nn.Linear(size//64, size//128), nn.ReLU(),)
        
        
        self.output_layer = nn.Linear(4*pose_time_dim, 1)
        
        self.softmax  = torch.nn.LogSoftmax(dim=-1)#dim=1)
        
    #pylint: disable=arguments-differ
    def forward(self,
                audio: Tensor,
                pose: Tensor,
                src_length: Tensor=None,
                mask: Tensor=None) -> (Tensor, Tensor):
        
        # B, N, D1 = audio.shape
        # B, M, D2 = pose.shape

        ##pose embed
        pose_embed =  self.trg_embed(pose)

        ## apply self attn to pose inputs
        y = pose_embed

        # Add position encoding
        y = self.pe(y)

        # Add Dropout
        y = self.emb_dropout(y)

        # # Apply each layer to the input
        for layer in self.pose_layers:
            y = layer(y)

        
        ##audio embed
        audio_embed = self.src_downsample(audio) # to make N=M
        
        x = audio_embed
        
        # Add position encoding 
        x = self.pe(x)

        # Add Dropout
        x = self.emb_dropout(x)

        # Apply each layer to the input
        for layer in self.aud_layers:
            x = layer(x)
        
        # fuse audio and pose embedding using cross attn
        for layer in self.fusion_layers:
            x = layer(x, y)

        # self attn
        x = self.self_attention(x)
            
        x = self.feed_forward(x)

        ## MLP for classification
        x = self.linear_layers(x)
        
        aud_vid_out = self.output_layer(x.view(x.shape[0],-1))  ## Bs, 1
        

        return aud_vid_out #.squeeze(1)       
        

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)
