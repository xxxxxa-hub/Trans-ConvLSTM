import torch
import torch.nn
from models.saconvlstm import *
import torch.nn.functional as F

class LSTMSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=13, hidden_dim=64, nclasses=8, kernel_size=(3,3), bias=False):
        super(LSTMSequentialEncoder, self).__init__()

        self.inconv = torch.nn.Conv3d(input_dim,hidden_dim,(1,3,3))
        self.cell = SAConvLSTMCell(input_dim,input_dim,input_dim,kernel_size)
        self.final = torch.nn.Conv2d(input_dim, nclasses, (1, 1))

    def forward(self, x1, hidden=None, state=None):

        # (b x t x c x h x w) -> (b x c x t x h x w)
        x1 = x1.permute(0,2,1,3,4)

        b, c, t, h, w = x1.shape
        hidden1 = torch.zeros(b, c, h, w).cuda()
        cell1 = torch.zeros(b, c, h, w).cuda()
        memory1 = torch.zeros(b, c, h, w).cuda()
        
        for iter in range(t):
            cell1,hidden1,memory1 = self.cell.forward(x1[:,:,iter,:,:], cell1,hidden1,memory1)

        
        x = self.final.forward(cell1)

        return F.log_softmax(x, dim=1)