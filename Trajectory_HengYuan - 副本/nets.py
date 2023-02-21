from layers import TrajBlock as TB
import torch.nn as nn
import tensorflow as tf
import torch

class TrajectoryCNN(nn.Module):
    def _init_weights(self,m):
        if isinstance(m,nn.Conv2d):
            print("Now initializing:"+str(m))
            nn.init.xavier_uniform(m.weight,gain = nn.init.calculate_gain('leaky_relu'))
    def __init__(self,keep_prob, seq_length, input_length, stacklength, num_hidden,filter_size):
        super(TrajectoryCNN, self).__init__()
        self.stacklength =stacklength
        self.keep_prob =keep_prob
        self.seq_length =seq_length
        self.filter_size=filter_size
        self.num_hidden=num_hidden

        self.conv1 = nn.Sequential(
            nn.Conv2d(seq_length,num_hidden[0],1,padding='same'),
            nn.LeakyReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(num_hidden[4], seq_length - input_length, filter_size, padding='same'),
            nn.LeakyReLU(),
        )
        self.decoder2 =nn.Conv2d(seq_length - input_length,seq_length - input_length,1,padding='same')
        self.TB1 = TB(filter_size, num_hidden, keep_prob, num_hidden[0])
        self.TB2 = TB(filter_size, num_hidden, keep_prob, num_hidden[0])
        self.TB3 = TB(filter_size, num_hidden, keep_prob, num_hidden[0])
        self.TB4 = TB(filter_size, num_hidden, keep_prob, num_hidden[0])
        self.apply(self._init_weights)

    def forward(self, images):
        seq_length=self.seq_length
        stacklength=self.stacklength
        filter_size=self.filter_size
        num_hidden=self.num_hidden
        keep_prob=self.keep_prob
        TB1 =self.TB1
        TB2 =self.TB2
        TB3=self.TB3
        TB4=self.TB4


        h = images[:, 0:seq_length, :, :]
        inputs = h
        out=[]
        inputs =self.conv1(inputs)
        inputs =TB1(inputs)
        inputs=TB2(inputs)
        inputs=TB3(inputs)
        inputs=TB4(inputs)
        out = self.decoder1(inputs)
        out =self.decoder2(out)
        gen_images = out

        return gen_images

