from lasagne.layers import Layer


class ForgetSizeLayer(Layer):

    '''
    This is a workaround to be able to broadcast in ElemwiseMergeLayer.
    It changes the output shape on the specified axis to None.
    '''

    def __init__(self, incoming, axis=-1, **kwargs):
        super(ForgetSizeLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input_, **kwargs):
        return input_

    def get_output_shape_for(self, input_shape, **kwargs):
        shape = list(input_shape)
        shape[self.axis] = None
        return tuple(shape)
