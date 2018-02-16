import theano.tensor as T
from lasagne.layers import MergeLayer


class MaskedSoftmaxLayer(MergeLayer):

    '''
    This layer performs row-wise softmax operation on a 2D tensor.
    Mask parameter specifies which parts of incoming tensor are parts
    of input, so that rows can contain sequences of different length.
    '''

    def __init__(self, incoming, mask, **kwargs):

        assert len(incoming.output_shape) == 2
        assert len(mask.output_shape) == 2

        super(MaskedSoftmaxLayer, self).__init__([incoming, mask], **kwargs)

    def get_output_for(self, inputs, **kwargs):

        assert len(inputs) == 2
        input_, mask = inputs

        input_max = input_.max(axis=1).dimshuffle(0, 'x')
        input_ = T.exp(input_ - input_max) * mask
        sums = input_.sum(axis=1).dimshuffle(0, 'x')
        return input_ / sums

    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 2
        return input_shapes[0]
