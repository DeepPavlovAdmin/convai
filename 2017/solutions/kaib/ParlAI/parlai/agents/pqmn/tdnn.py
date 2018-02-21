import torch as t
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

#import pdb

class TDNN(nn.Module):
    def __init__(self, opt):
        super(TDNN, self).__init__()
        self.opt = opt
        #pdb.set_trace()
        self.kernels = [Parameter(t.Tensor(out_dim, opt['embedding_dim_char'], kW).uniform_(-1, 1))
                        for kW, out_dim in opt['kernels']]
        self._add_to_parameters(self.kernels, 'TDNN_kernel')

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, max_seq_len, max_word_len, char_embed_size]

        :return: tensor with shape [batch_size, max_seq_len, depth_sum]

        applies multikenrel 1d-conv layer along every word in input with max-over-time pooling
            to emit fixed-size output
        """

        input_size = x.size()
        input_size_len = len(input_size)

        assert input_size_len == 4, \
            'Wrong input rang, must be equal to 4, but {} found'.format(input_size_len)

        [batch_size, seq_len, _, embed_size] = input_size

        assert embed_size == self.opt['embedding_dim_char'], \
            'Wrong embedding size, must be equal to {}, but {} found'.format(self.opt['embedding_dim_char'], embed_size)

        # leaps with shape
        x = x.view(-1, self.opt['max_word_len'], self.opt['embedding_dim_char']).transpose(1, 2).contiguous()

        xs = [F.tanh(F.conv1d(x, kernel)) for kernel in self.kernels]
        xs = [x.max(2)[0].squeeze(2) for x in xs]

        x = t.cat(xs, 1)
        x = x.view(batch_size, seq_len, -1)

        return x

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)