# as in https://arxiv.org/abs/1703.04816

import theano.tensor as T
from lasagne.layers import MergeLayer
from lasagne import init


class WeightedFeatureLayer(MergeLayer):

    '''
    A layer calculating weighted word-in-question features from FastQA.

    Parameters:
        incomings: [contexts, questions, context_mask, question_mask]
        V:         initializer for the weight vector, defaults to Uniform()

    Returns:
        a layer of shape (batch_size, context_len) containing the features
    '''

    def __init__(self, incomings, V=init.Uniform(), **kwargs):

        assert len(incomings) == 4
        assert len(incomings[0].output_shape) == 3
        assert len(incomings[1].output_shape) == 3
        assert len(incomings[2].output_shape) == 2
        assert len(incomings[3].output_shape) == 2

        super(WeightedFeatureLayer, self).__init__(incomings, **kwargs)

        emb_size = incomings[0].output_shape[2]
        self.V = self.add_param(V, (emb_size,), name="V")

    def get_output_for(self, inputs, **kwargs):

        assert len(inputs) == 4

        context, question, c_mask, q_mask = inputs
        batch_size, question_len, emb_size = question.shape

        question = question.reshape(
            (batch_size * question_len, emb_size)) * self.V
        question = question.reshape((batch_size, question_len, emb_size))

        # batch_size x emb_size x context_len
        context = context.dimshuffle(0, 2, 1)

        # batch_size x question_len x context_len
        x = T.batched_dot(question, context)
        x_max = x.max(axis=2).dimshuffle(0, 1, 'x')
        esim = T.exp(x - x_max)
        esim *= c_mask.reshape((batch_size, 1, -1))

        sums = esim.sum(axis=2)
        esim /= sums.dimshuffle(0, 1, 'x')

        esim *= q_mask.reshape((batch_size, -1, 1))

        return esim.sum(axis=1)  # batch_size x context_len

    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 4
        return input_shapes[0][:2]
