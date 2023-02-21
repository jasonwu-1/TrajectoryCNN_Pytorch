import torch
import torch.nn as nn

class TrajBlock(nn.Module):

    def __init__(self, filter_size,num_hidden,keep_prob,in_channels):
        super(TrajBlock,self).__init__()
        self.filter_size = filter_size
        self.num_hidden = num_hidden
        self.keep_prob = keep_prob
        self.in_channels = in_channels

        self.h0 = nn.Conv2d(in_channels,num_hidden[0],1,padding='same')
        self.leakyRelu = nn.LeakyReLU()
        self.traj1 =nn.Conv2d(in_channels,num_hidden[0],filter_size,padding='same')
        self.dropout = nn.Dropout(keep_prob)
        self.h1 = nn.Conv2d(num_hidden[0],num_hidden[1],1,padding='same')
        self.traj2 = nn.Conv2d(num_hidden[0],num_hidden[1],filter_size,padding='same')
        self.h2 =nn.Conv2d(num_hidden[1],num_hidden[2],1,padding='same')
        self.traj3 = nn.Conv2d(num_hidden[1],num_hidden[2],filter_size,padding='same')
        self.traj4 = nn.Conv2d(num_hidden[2],num_hidden[3],filter_size,padding='same')
        self.traj5 = nn.Conv2d(num_hidden[3],num_hidden[4],filter_size,padding='same')


    def forward(self,x):
        h0_x= self.h0(x)
        h0_x = self.leakyRelu(h0_x)

        traj1_x =self.traj1(x)
        traj1_x = self.leakyRelu(traj1_x)
        traj1_x = self.dropout(traj1_x)

        h1_x =self.h1(traj1_x)
        h1_x =self.leakyRelu(h1_x)

        traj2_x = self.traj2(traj1_x)
        traj2_x =self.leakyRelu(traj2_x)
        traj2_x=self.dropout(traj2_x)

        h2_x =self.h2(traj2_x)
        h2_x = self.leakyRelu(h2_x)

        traj3_x =self.traj3(traj2_x)
        traj3_x =self.leakyRelu(traj3_x)
        traj3_x =self.dropout(traj3_x)

        traj4_x =self.traj4(traj3_x+h2_x)
        traj4_x =self.leakyRelu(traj4_x)
        traj4_x =self.dropout(traj4_x)

        traj5_x =self.traj5(traj4_x+h1_x)
        traj5_x=self.leakyRelu(traj5_x)
        traj5_x=self.dropout(traj5_x)

        return  traj5_x+h0_x




