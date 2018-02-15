import theano.tensor as T
from lasagne import init
from lasagne.layers import Layer


class HighwayLayer(Layer):

    '''
    A Highway Layer (https://arxiv.org/pdf/1505.00387.pdf).
    Some part of an input passes through the layer unchanged.
    Output shape is the same as input shape.
    '''

    def __init__(self, incoming,
                 W1=init.GlorotUniform(),
                 b1=init.Constant(0.),
                 W2=init.GlorotUniform(),
                 b2=init.Constant(0.),
                 **kwargs):

        assert len(incoming.output_shape) == 2

        super(HighwayLayer, self).__init__(incoming, **kwargs)

        n_inputs = incoming.output_shape[1]

        self.W1 = self.add_param(W1, (n_inputs, n_inputs), name="W1")
        self.b1 = self.add_param(b1, (n_inputs,), name="b1",
                                 regularizable=False)
        self.W2 = self.add_param(W2, (n_inputs, n_inputs), name="W2")
        self.b2 = self.add_param(b2, (n_inputs,), name="b2",
                                 regularizable=False)

    def get_output_for(self, input_, **kwargs):
        g = T.nnet.sigmoid(input_.dot(self.W1) + self.b1)
        input_changed = T.tanh(input_.dot(self.W2) + self.b2)
        return g * input_ + (1. - g) * input_changed

    def get_output_shape_for(self, input_shape, **kwargs):
        return input_shape
