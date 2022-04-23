from models.transunet import *
import torch

class Model(nn.Module):
    def __init__(self,img_dim=48,in_channels=3,out_channels=64,head_num=4,mlp_dim=512,block_num=6,patch_dim=16,class_num=18,height=48, width=48, input_dim=18, hidden_dim=64, nclasses=8, kernel_size=(3,3), bias=False):
        super(Model,self).__init__()
        self.transunet = TransUNet(img_dim=48,
                          in_channels=13,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=6,
                          patch_dim=8,
                          class_num=18)
        
        self.lstm = LSTMSequentialEncoder(48,48,input_dim=16,nclasses=18)
        self.lstm = torch.nn.DataParallel(self.lstm).cuda()
    
    def forward(self,input,target):
        list = []
        for i in range(30):
            list.append(self.transunet(input[:,i,:,:,:]).unsqueeze(1))
        input1 = torch.cat(list,dim=1)
        input1 = input1.cuda()
        target = target.cuda()
        
        return self.lstm(input1),target
