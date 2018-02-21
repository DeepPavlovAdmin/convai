import theano.tensor as TT
from lasagne.layers import MergeLayer
from lasagne import init


class TrainNAWLayer(MergeLayer):

    '''
    A layer for training NAW token (for negative training).

    Parameters:
        incoming:    incoming 3D layer (batch_size x seq_len x emb_size)
        mask:        lasagne layer, 2D binary mask (batch_size x seq_len)
        output_size: embedding size
        W:           initializer for NAW vector, defaults to Normal()

    Return:
        incoming layer with every last vector in a sequence replaced by W
    '''

    def __init__(self, incoming, mask, output_size, W=init.Normal(), **kwargs):

        super(TrainNAWLayer, self).__init__([incoming, mask], **kwargs)

        self.output_size = output_size

        self.W = self.add_param(W, (output_size,), name="W")

    def get_output_for(self, inputs, **kwargs):
        assert len(inputs) == 2
        input_, mask = inputs
        naw_positions = mask.astype('int32').sum(axis=1) - 1
        return TT.set_subtensor(
            input_[TT.arange(input_.shape[0]), naw_positions], self.W)

    def get_output_shape_for(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]
