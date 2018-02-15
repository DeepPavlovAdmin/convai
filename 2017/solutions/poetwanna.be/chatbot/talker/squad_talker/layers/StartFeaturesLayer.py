import theano.tensor as T
from lasagne.layers import MergeLayer


class StartFeaturesLayer(MergeLayer):

    '''
    This layer prepares the input to equation (8) in FastQA paper.
    '''

    def __init__(self, incomings, **kwargs):

        assert len(incomings) == 2
        assert len(incomings[0].output_shape) == 3
        assert len(incomings[1].output_shape) == 2

        super(StartFeaturesLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):

        assert len(inputs) == 2
        H, z = inputs
        z_tiled = T.tile(z.dimshuffle(0, 'x', 1), (H.shape[1], 1))
        return T.concatenate([H, z_tiled, H * z_tiled], axis=2)

    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 2
        shape = list(input_shapes[0])
        shape[2] *= 3
        return tuple(shape)
