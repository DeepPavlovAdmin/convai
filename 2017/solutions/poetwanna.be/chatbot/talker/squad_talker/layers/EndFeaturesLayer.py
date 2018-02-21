import theano.tensor as T
from lasagne.layers import MergeLayer


class EndFeaturesLayer(MergeLayer):

    '''
    This layer prepares the input to equation (9) in FastQA paper.
    '''

    def __init__(self, incomings, **kwargs):

        assert len(incomings) == 3
        assert len(incomings[0].output_shape) == 3
        assert len(incomings[1].output_shape) == 2
        assert len(incomings[2].output_shape) == 1

        super(EndFeaturesLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):

        assert len(inputs) == 3
        H, z, start_inds = inputs
        z_tiled = T.tile(z.dimshuffle(0, 'x', 1), (H.shape[1], 1))
        h_s = H[T.arange(H.shape[0]), start_inds]
        h_s_tiled = T.tile(h_s.dimshuffle(0, 'x', 1), (H.shape[1], 1))

        return T.concatenate(
            [H, h_s_tiled, z_tiled, H * z_tiled, H * h_s_tiled], axis=2)

    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 3
        shape = list(input_shapes[0])
        shape[2] *= 5
        return tuple(shape)
