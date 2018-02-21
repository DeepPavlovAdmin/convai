import theano.tensor as TT
from lasagne.layers import MergeLayer
from lasagne import init


class TrainUnkLayer(MergeLayer):

    '''
    A layer for training one specific word embedding, later called UNK.

    Parameters:
        incoming:    incoming 3D layer (batch_size x seq_len x emb_size)
        mask:        lasagne layer, 2D binary mask showing UNK positions
        output_size: embedding size
        W:           initializer for UNK vector, defaults to Normal()

    Return:
        incoming layer with every instance of UNK replaced by W
    '''

    def __init__(self, incoming, mask, output_size, W=init.Normal(), **kwargs):

        super(TrainUnkLayer, self).__init__([incoming, mask], **kwargs)

        self.output_size = output_size

        self.W = self.add_param(W, (output_size,), name="W")

    def get_output_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        input_, mask = inputs
        return TT.set_subtensor(input_[mask.nonzero()], self.W)

    def get_output_shape_for(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]
