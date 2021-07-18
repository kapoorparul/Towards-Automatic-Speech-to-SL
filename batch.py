# coding: utf-8

"""
Implementation of a mini-batch.
"""

import torch
import torch.nn.functional as F

from constants import TARGET_PAD

def gen_mask(src, src_lengths):
    src_shp = src.shape
    
    mask = torch.ones((src_shp[0], src_shp[1]),  dtype=torch.bool)
    
    for i, inp in enumerate(src):
        for ii, j in enumerate(inp):
            if ii >= src_lengths[i]:
                mask[i][ii] = 0
    
    return mask.unsqueeze(1)




class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, nonreg_pad_index, model):

        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = torch_batch.src
        
        self.src_mask = gen_mask(self.src, self.src_lengths)#(self.src != pad_index).unsqueeze(1)
        
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        self.file_paths = torch_batch.file_paths
        self.use_cuda = model.use_cuda
        self.target_pad = TARGET_PAD
        
        self.nonreg_trg, self.nonreg_trg_lengths = torch_batch.nonreg_trg
        self.nonreg_trg_inp = self.nonreg_trg[:,:-1]
        self.nonreg_trg_mask = (self.nonreg_trg_inp != nonreg_pad_index).unsqueeze(1)
        self.nonreg_trg = self.nonreg_trg[:,1:]
        
        # Just Count
        self.just_count_in = model.just_count_in
        
        # Future Prediction
        self.future_prediction = model.future_prediction

        if hasattr(torch_batch, "trg"):
            trg = torch_batch.trg
            trg_lengths = torch_batch.trg.shape[1]
            # trg_input is used for teacher forcing, last one is cut off
            # Remove the last frame for target input, as inputs are only up to frame N-1
            self.trg_input = trg.clone()[:, :-1,:]

            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg.clone()[:, 1:, :]

            # Just Count
            if self.just_count_in:
                # If Just Count, cut off the first frame of trg_input
                self.trg_input = self.trg_input[:, :, -1:]

            # Future Prediction
            if self.future_prediction != 0:
                # Loop through the future prediction, concatenating the frames shifted across once each time
                future_trg = torch.Tensor()
                # Concatenate each frame (Not counter)
                for i in range(0, self.future_prediction):
                    future_trg = torch.cat((future_trg, self.trg[:, i:-(self.future_prediction - i), :-1].clone()), dim=2)
                # Create the final target using the collected future_trg and original trg
                self.trg = torch.cat((future_trg, self.trg[:,:-self.future_prediction,-1:]), dim=2)

                # Cut off the last N frames of the trg_input
                self.trg_input = self.trg_input[:, :-self.future_prediction, :]

            # Target Pad is dynamic, so we exclude the padded areas from the loss computation
            trg_mask = (self.trg_input != self.target_pad).unsqueeze(1)
            # This increases the shape of the target mask to be even (16,1,120,120) -
            # adding padding that replicates - so just continues the False's or True's
            pad_amount = self.trg_input.shape[1] - self.trg_input.shape[2]
            # Create the target mask the same size as target input
            self.trg_mask = (F.pad(input=trg_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
            self.ntokens = 0#(self.trg != pad_index).data.sum().item()

        if self.use_cuda:
            self._make_cuda()

    # If using Cuda
    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()
        if self.nonreg_trg is not None:
            self.nonreg_trg_inp = self.nonreg_trg_inp.cuda()
            self.nonreg_trg_mask = self.nonreg_trg_mask.cuda()
            self.nonreg_trg = self.nonreg_trg.cuda()

