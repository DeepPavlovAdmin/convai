from collections import OrderedDict
import cPickle
import os

def prototype_state():
    state = {}

    # ----- CONSTANTS -----
    # Random seed
    state['seed'] = 1234
    
    # Logging level
    state['level'] = 'DEBUG'

    # Out-of-vocabulary token string
    state['oov'] = '<unk>'
    
    # These are end-of-sequence marks
    state['end_sym_utterance'] = '</s>'

    # Special tokens need to be defined here, because model architecture may adapt depending on these
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = 2 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = 3 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = 4 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = 5 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = 6 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = 7 # voice over symbol <voice_over>
    state['off_screen_sym'] = 8 # off screen symbol <off_screen>
    state['pause_sym'] = 9 # pause symbol <pause>


    # ----- MODEL ARCHITECTURE -----
    # If this flag is on, the hidden state between RNNs in subsequences is always initialized to zero.
    # Set this to reset all RNN hidden states between 'max_grad_steps' time steps
    state['reset_hidden_states_between_subsequences'] = False

    # If this flag is on, the maxout activation function will be applied to the utterance decoders output unit.
    # This requires qdim_decoder = 2x rankdim
    state['maxout_out'] = False

    # If this flag is on, a one-layer MLP with linear activation function will applied
    # on the utterance decoder hidden state before outputting the distribution over words.
    state['deep_utterance_decoder_out'] = True

    # If this flag is on, there will be an extra MLP between utterance and dialogue encoder
    state['deep_dialogue_encoder_input'] = False

    # Default and recommended setting is: tanh.
    # The utterance encoder and utterance decoder activation function
    state['sent_rec_activation'] = 'lambda x: T.tanh(x)'
    # The dialogue encoder activation function
    state['dialogue_rec_activation'] = 'lambda x: T.tanh(x)'
    
    # Determines how to input the utterance encoder and dialogue encoder into the utterance decoder RNN hidden state:
    #  - 'first': initializes first hidden state of decoder using encoders
    #  - 'all': initializes first hidden state of decoder using encoders, 
    #            and inputs all hidden states of decoder using encoders
    #  - 'selective': initializes first hidden state of decoder using encoders, 
    #                 and inputs all hidden states of decoder using encoders.
    #                 Furthermore, a gating function is applied to the encoder input 
    #                 to turn off certain dimensions if necessary.
    #
    # Experiments show that 'all' is most effective.
    state['decoder_bias_type'] = 'all' 

    # Define the gating function for the three RNNs.
    state['utterance_encoder_gating'] = 'GRU' # Supports 'None' and 'GRU'
    state['dialogue_encoder_gating'] = 'GRU' # Supports 'None' and 'GRU'
    state['utterance_decoder_gating'] = 'GRU' # Supports 'None', 'BOW' (Bag of Words), 'GRU' and 'LSTM'

    # If this flag is on, two utterances encoders (one forward and one backward) will be used,
    # otherwise only a forward utterance encoder is used.
    state['bidirectional_utterance_encoder'] = False

    # If this flag is on, there will be a direct connection between utterance encoder and utterance decoder RNNs.
    state['direct_connection_between_encoders_and_decoder'] = False

    # If this flag is on, there will be an extra MLP between utterance encoder and utterance decoder.
    state['deep_direct_connection'] = False

    # If the 'direct_connection_between_encoders_and_decoder' is on, then enabling this flag will
    # change the model so that it does not use the dialogue encoder (context encoder)
    state['disable_dialogue_encoder'] = False


    # If this flag is on, the model will collaps to a standard RNN:
    # 1) The utterance+dialogue encoder input to the utterance decoder will be zero
    # 2) The utterance decoder will never be reset
    # Note this model will always be initialized with a hidden state equal to zero.
    state['collaps_to_standard_rnn'] = False

    # If this flag is on, the utterance decoder will be reset after each end-of-utterance token.
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    # If this flag is on, the utterance encoder will be reset after each end-of-utterance token.
    state['reset_utterance_encoder_at_end_of_utterance'] = False


    # ----- HIDDEN LAYER DIMENSIONS -----
    # Dimensionality of (word-level) utterance encoder hidden state
    state['qdim_encoder'] = 512
    # Dimensionality of (word-level) utterance decoder (RNN which generates output) hidden state
    state['qdim_decoder'] = 512
    # Dimensionality of (utterance-level) context encoder hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank word embedding approximation
    state['rankdim'] = 256


    # ----- LATENT VARIABLES WITH VARIATIONAL LEARNING -----
    # If this flag is on, a Gaussian latent variable is added at the beginning of each utterance.
    # The utterance decoder will be conditioned on this latent variable,
    # and the model will be trained using the variational lower bound. 
    # See, for example, the variational auto-encoder by Kingma et al. (2013).
    state['add_latent_gaussian_per_utterance'] = False

    # This flag will condition the latent variables on the dialogue encoder
    state['condition_latent_variable_on_dialogue_encoder'] = False
    # This flag will condition the latent variable on the DCGM (mean pooling over words) encoder.
    # This will replace the conditioning on the utterance encoder.
    # If the flag is false, the latent variable will be conditioned on the utterance encoder RNN.
    state['condition_posterior_latent_variable_on_dcgm_encoder'] = False
    # Dimensionality of Gaussian latent variable, which has diagonal covariance matrix.
    state['latent_gaussian_per_utterance_dim'] = 10

    # This is a constant by which the diagonal covariance matrix is scaled.
    # By setting it to a high number (e.g. 1 or 10),
    # the KL divergence will be relatively low at the beginning of training.
    state['scale_latent_gaussian_variable_variances'] = 10
    state['min_latent_gaussian_variable_variances'] = 0.01
    state['max_latent_gaussian_variable_variances'] = 10.0





    # If this flag is on, the utterance decoder will ONLY be conditioned on the Gaussian latent variable.
    state['condition_decoder_only_on_latent_variable'] = False


    # If this flag is on, a piecewise latent variable is added at the beginning of each utterance.
    # The utterance decoder will be conditioned on this latent variable,
    # and the model will be trained using the variational lower bound. 
    # See, for example, the variational auto-encoder by Kingma et al. (2013).
    state['add_latent_piecewise_per_utterance'] = False

    # If this flag is on, the posterior piecewise distribution will be interpolated
    # with the prior distribution using a linear gating mechanism.
    state['gate_latent_piecewise_per_utterance'] = True


    state['latent_piecewise_alpha_variables'] = 5

    # This is a constant by which the prior piecewise alpha parameters are scaled.
    # By setting it to a number in the range (2.0, 10) the piecewise posterior distributions will
    # be free to change appropriately to accomodate the real posterior,
    # while still leaving some probability mass around 0.5 for the variable to change.
    # With scale_latent_piecewise_variable_alpha=10, KL divergence cost is about 10% of overall cost initially.
    # With scale_latent_piecewise_variable_alpha=1, KL divergence cost is about 1% of overall cost initially.

    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True

    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0


    state['latent_piecewise_per_utterance_dim'] = 10

    # If parameter tying is enabled, a Gaussian convolution is applied to all the the alpha values.
    # This makes the alpha values dependent upon each other, and guarantees that a single sample
    # will update the weight of all the alpha values with higher gradients to nearby values.
    # Julian: This only helped slightly in my intial experiments.
    state['latent_piecewise_variable_alpha_parameter_tying'] = False
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 1.0

    # If this flag is on, the input to the utterance decoder will be passed through
    # a one-layer MLP with rectified linear units.
    # If batch normalization or layer normalization is on,
    # this will also ensure that the inputs to the decoder RNN are normalized.
    state['deep_utterance_decoder_input'] = True


    # If this flag is on, the KL-divergence term weight for the latent variables
    # will be slowly increased from zero to one.
    state['train_latent_variables_with_kl_divergence_annealing'] = False

    # The KL-divergence term weight is increased by this parameter for every training batch.
    # It is truncated to one. For example, 1.0/60000.0 means that at iteration 60000 the model
    # will assign weight one to the KL-divergence term
    # and thus only be maximizing the true variational bound from iteration 60000 and onward.
    state['kl_divergence_annealing_rate'] = 1.0/60000.0

    # If this flag is enabled, previous token input to the decoder RNN is replaced with 'unk' tokens at random.
    state['decoder_drop_previous_input_tokens'] = False
    # The rate at which the previous tokesn input to the decoder is kept (not set to 'unk').
    # Setting this to zero effectively disables teacher-forcing in the model.
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    # If this flag is enabled, mean field inference with stochastic gradient descent is applied during test time.
    # Julian: This didn't really make a big difference...
    state['apply_meanfield_inference'] = False

    # Word embedding initialization
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = ''
    state['fix_pretrained_word_embeddings'] = False

    # If this flag is on, the model will fix the parameters of the utterance encoder and dialogue encoder RNNs,
    # as well as the word embeddings. NOTE: NOT APPLICABLE when the flag 'collaps_to_standard_rnn' is on.
    state['fix_encoder_parameters'] = False

    # If this flag is disabled, the model will not generate the first utterance in a dialogue.
    # This is used for the debate dataset as well as the skip_utterance configuration.
    state['do_generate_first_utterance'] = False

    # If this flag is enabled, the data iterator is changed so that the model is conditioned 
    # on exactly one utterance and predicts only one utterance; the utterance to predict is
    # either the next utterance or the previous utterance in the dialogue.
    # When this flag is on, it forces the 'do_generate_first_utterance' to be off.
    state['skip_utterance'] = False

    # If 'skip_utterance' flag is enabled together with this flag, the data iterator is changed so
    # that the model always predicts both the previous and next utterances.
    # Note, this will double the batch size!
    state['skip_utterance_predict_both'] = False


    # ----- TRAINING PROCEDURE -----
    # Choose optimization algorithm (adam works well most of the time)
    state['updater'] = 'adam'
    # If this flag is on, NCE (Noise-Contrastive Estimation) will be used to train model.
    # This is significantly faster for large vocabularies (e.g. more than 20K words), 
    # but experiments show that this degrades performance.
    state['use_nce'] = False
    # Threshold to clip the gradient
    state['cutoff'] = 0.01
    # Learning rate. The rate 0.0002 seems to work well across many tasks with adam.
    # Alternatively, the learning rate can be adjusted down (e.g. 0.00004) 
    # to at the end of training to help the model converge well.
    state['lr'] = 0.0002
    # Early stopping configuration
    state['patience'] = 20
    state['cost_threshold'] = 1.003
    # Batch size. If out of memory, modify this!
    state['bs'] = 80
    # Sort by length groups of  
    state['sort_k_batches'] = 20
    # Training examples will be split into subsequences.
    # This parameter controls the maximum size of each subsequence.
    # Gradients will be computed on the subsequence, and the last hidden state of all RNNs will
    # be used to initialize the hidden state of the RNNs in the next subsequence.
    state['max_grad_steps'] = 80
    # Modify this in the prototype
    state['save_dir'] = './'
    # Frequency of training error reports (in number of batches)
    state['train_freq'] = 10
    # Validation frequency
    state['valid_freq'] = 5000
    # Number of batches to process
    state['loop_iters'] = 3000000
    # Maximum number of minutes to run
    state['time_stop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1
    # Maximum dialogue length
    state['max_len'] = -1

    # The model can apply several normalization operators to the encoder hidden states:
    # 'NONE': No normalization is applied.
    # 'BN': Batch normalization is applied.
    # 'LN': Layer normalization is applied.
    #
    # Note the normalization operators can only be applied to GRU encoders and feed-forward neural networks.
    state['normop_type'] = 'LN'

    if state['normop_type'] == 'BN':
        state['normop_gamma_init'] = 0.1
        state['normop_gamma_min'] = 0.05
        state['normop_gamma_max'] = 10.0
        state['normop_moving_average_const'] = 0.99
        state['normop_max_enc_seq'] = 50
    else:
        state['normop_gamma_init'] = 1.0
        state['normop_gamma_min'] = 0.05
        state['normop_gamma_max'] = 10.0
        state['normop_moving_average_const'] = 0.99
        state['normop_max_enc_seq'] = 1

    # Parameters for initializing the training data iterator.
    # The first is the first offset position in the list examples.
    # The second is the number of reshuffles to perform at the beginning.
    state['train_iterator_offset'] = 0
    state['train_iterator_reshuffle_count'] = 1

    state['use_pg'] = False
    return state

def prototype_test():
    state = prototype_state()
    
    # Fill paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttest.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/tvalid.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    state['max_grad_steps'] = 20
    
    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = False
    
    state['valid_freq'] = 50
    
    state['prefix'] = "testmodel_" 
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_utterance_decoder_out'] = True
    state['deep_dialogue_encoder_input'] = True

    state['utterance_encoder_gating'] = 'GRU'
    state['dialogue_encoder_gating'] = 'GRU'
    state['utterance_decoder_gating'] = 'GRU'
    state['bidirectional_utterance_encoder'] = True 
    state['direct_connection_between_encoders_and_decoder'] = True

    state['bs'] = 5
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all'
    
    state['qdim_encoder'] = 15
    state['qdim_decoder'] = 5
    state['sdim'] = 10
    state['rankdim'] = 10



    return state

def prototype_test_variational():
    state = prototype_state()
    
    # Fill paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttest.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/tvalid.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    state['max_grad_steps'] = 20

    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    
    state['valid_freq'] = 5
   
    state['prefix'] = "testmodel_"
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_utterance_decoder_out'] = True
    state['deep_dialogue_encoder_input'] = True
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['utterance_encoder_gating'] = 'GRU'
    state['dialogue_encoder_gating'] = 'GRU'
    state['utterance_decoder_gating'] = 'LSTM'

    state['bidirectional_utterance_encoder'] = False

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 5
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['condition_posterior_latent_variable_on_dcgm_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 10

    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['bs'] = 5
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all'
    
    state['qdim_encoder'] = 15
    state['qdim_decoder'] = 5
    state['sdim'] = 10
    state['rankdim'] = 10

    state['gate_latent_piecewise_per_utterance'] = False

    # KL max-trick
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['max_kl_percentage'] = 0.01

    return state


# Twitter LSTM RNNLM model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016), with batch norm / layer norm extension.
def prototype_twitter_lstm():
    state = prototype_state()
    
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    state['max_grad_steps'] = 80
    
    state['valid_freq'] = 5000
    
    state['prefix'] = "TwitterModel_" 
    state['updater'] = 'adam'
    
    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['collaps_to_standard_rnn'] = True
 
    state['bs'] = 80 
    state['decoder_bias_type'] = 'all'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    state['sdim'] = 10
    state['rankdim'] = 400

    return state




def prototype_twitter_HRED():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    return state

def prototype_reddit_HRED():
    state = prototype_state()
 
    # Fill your paths here!
    state['train_dialogues'] = "/home/ml/mnosew1/data/reddit/Training.dialogues.pkl"
    state['test_dialogues'] = "/home/ml/mnosew1/data/reddit/Test.dialogues.pkl"
    state['valid_dialogues'] = "/home/ml/mnosew1/data/reddit/Validation.dialogues.pkl"
    state['dictionary'] = "/home/ml/mnosew1/data/reddit/Training.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "RedditHRED_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 2000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    return state


# Twitter HRED model, where context biases decoder using standard MLP.
def prototype_twitter_HRED_StandardBias():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    return state



def prototype_twitter_VHRED():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True



    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'


    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100


    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


# Twitter VHRED model, where context biases decoder using standard MLP.
# Note, this model should be pretrained as HRED model.
def prototype_twitter_VHRED_StandardBias():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True



    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'


    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100


    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


# Ubuntu LSTM RNNLM model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016), with batch norm / layer norm extension.
def prototype_ubuntu_LSTM():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['collaps_to_standard_rnn'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    state['sdim'] = 10
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM' # Supports 'None', 'GRU' and 'LSTM'

    return state



def prototype_ubuntu_HRED():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 1000
    state['rankdim'] = 300

    return state



def prototype_ubuntu_VHRED():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 1000
    state['rankdim'] = 300

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


### New experiments

# Large Twitter HRED model.
def prototype_twitter_HRED_Large():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100

    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state




# Large Twitter HRED model.
def prototype_twitter_VHRED_Large_SkipConnections():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100

    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



# Large Twitter VHRED model.
def prototype_twitter_GaussPiecewise_VHRED_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_Gauss_VHRED_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state




def prototype_twitter_HRED_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 10
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 10

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state




### CLUSTER EXPERIMENTS BEGIN: BAG-OF-WORDS DECODER

def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
    state['scale_latent_piecewise_variable_prior_alpha'] = 0.1
    state['scale_latent_piecewise_variable_posterior_alpha'] = 0.1

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
    state['scale_latent_piecewise_variable_prior_alpha'] = 10.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 10.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 0.1
    state['scale_latent_piecewise_variable_posterior_alpha'] = 0.1

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp6():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 10.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 10.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp7():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp8():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp9():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 2
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp10():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp11():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp12():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 10
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp13():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.1

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp14():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 1.0

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp15():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 10.0

    state['patience'] = 20

    return state

def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp16():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 100.0

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp17():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.01

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp18():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.001

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp19():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 10
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.01

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp20():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 10
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.001

    state['patience'] = 20

    return state





### CLUSTER EXPERIMENTS END: BAG-OF-WORDS DECODER



### CLUSTER EXPERIMENTS BEGIN: Twitter Baselines

###
### We do hyperparameter search for LSTM:
###
# qdim_decoder = {1000, 2000, 4000}
# rankdim = 400
# lr = 0.0002
# bs = 80
# normop_type = 'LN'

def prototype_twitter_LSTM_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 1000
    state['sdim'] = 10
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['collaps_to_standard_rnn'] = True
    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False


    state['patience'] = 20

    return state


def prototype_twitter_LSTM_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    state['sdim'] = 10
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['collaps_to_standard_rnn'] = True
    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False

    state['patience'] = 20

    return state


def prototype_twitter_LSTM_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 4000
    state['sdim'] = 10
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['collaps_to_standard_rnn'] = True
    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False

    state['patience'] = 20

    return state





###
### We do hyperparameter search for HRED:
###
# sdim = {500, 1000}
# qdim_encoder = {1000}
# qdim_decoder = {1000, 2000, 4000}
# rankdim = 400
# bidirectional_utterance_encoder = True
# reset_utterance_encoder_at_end_of_utterance = False
# reset_utterance_decoder_at_end_of_utterance = True
# lr = 0.0002
# bs = 80
# normop_type = 'LN'

def prototype_twitter_HRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_HRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_HRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_HRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_HRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 2000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state

###
### We do hyperparameter search for Gaussian VHRED:
###
# sdim = {500, 1000}
# qdim_encoder = {1000}
# qdim_decoder = {1000, 2000, 4000}
# rankdim = 400
# latent_gaussian_per_utterance_dim = {100, 300}
# bidirectional_utterance_encoder = True
# reset_utterance_encoder_at_end_of_utterance = False
# reset_utterance_decoder_at_end_of_utterance = True
# lr = 0.0002
# bs = 80
# normop_type = 'LN'

def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state


def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state


def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state



def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state

def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state

###
### We do hyperparameter search for Piecewise-Gaussian VHRED:
###
# sdim = {500, 1000}
# qdim_encoder = {1000}
# qdim_decoder = {1000, 2000, 4000}
# rankdim = 400
# latent_gaussian_per_utterance_dim = {100, 300}
# latent_piecewise_per_utterance_dim = {100, 300}
# gate_latent_piecewise_per_utterance = {False, True}
# bidirectional_utterance_encoder = True
# reset_utterance_encoder_at_end_of_utterance = False
# reset_utterance_decoder_at_end_of_utterance = True
# lr = 0.0002
# bs = 80
# normop_type = 'LN'

def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 300
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp6():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    state['gate_latent_piecewise_per_utterance'] = False

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp7():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    state['gate_latent_piecewise_per_utterance'] = False

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp8():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 300
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    state['gate_latent_piecewise_per_utterance'] = False

    return state


### CLUSTER EXPERIMENTS END: Twitter Baselines




def prototype_twitter_Gauss_VHRED_LSTMDecoder_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_LSTMDecoder_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state






### CLUSTER EXPERIMENTS BEGIN: Ubuntu HRED Baseline

### To be executed on the Guillumin Cluster...

###
### We do hyperparameter search for HRED without activity-entity features:
###
# sdim = {500, 1000}
# qdim_encoder = {500, 1000}
# qdim_decoder = {1000, 2000}
# rankdim = 400
# bidirectional_utterance_encoder = True
# reset_utterance_encoder_at_end_of_utterance = False
# reset_utterance_decoder_at_end_of_utterance = True
# lr = 0.0002
# bs = 80
# normop_type = {'LN', 'NONE'}



def prototype_ubuntu_HRED_NormOp_ClusterExp1():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"

    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_ubuntu_HRED_NormOp_ClusterExp2():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"


    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_ubuntu_HRED_NormOp_ClusterExp3():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"


    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state

def prototype_ubuntu_HRED_NormOp_ClusterExp4():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"

    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['normop_type'] = 'NONE'

    state['patience'] = 20

    return state



def prototype_ubuntu_HRED_NormOp_ClusterExp5():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"


    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['normop_type'] = 'NONE'

    state['patience'] = 20

    return state



def prototype_ubuntu_HRED_NormOp_ClusterExp6():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"


    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['normop_type'] = 'NONE'

    state['patience'] = 20

    return state

### CLUSTER EXPERIMENTS END: Ubuntu HRED Baseline


def prototype_twitter_HRED_NoNormalization_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['normop_type'] = 'NONE'

    state['patience'] = 20

    return state





###
### We do hyperparameter search for HRED on CreateDebate:
###
### sdim = {500, 1000}
### qdim_encoder = {500, 1000}
### qdim_decoder = {500, 1000}
### rankdim = 300
### bidirectional_utterance_encoder = False
### reset_utterance_encoder_at_end_of_utterance = False
### reset_utterance_decoder_at_end_of_utterance = True
### lr = 0.0002
### bs = 80
### normop_type = 'LN'

def prototype_debate_HRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 500
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_HRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 500
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 50
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 50
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/5000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/5000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 300
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state





def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 500
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 50
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 50
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/5000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/5000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 0.1
    state['scale_latent_piecewise_variable_posterior_alpha'] = 0.1

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 0.1
    state['scale_latent_piecewise_variable_posterior_alpha'] = 0.1

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.50

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state






def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp6():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/90000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp7():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 300
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp8():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.01


    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp9():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.025

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp10():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.05

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp11():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.005

    return state




def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.01

    return state


def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp6():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.025

    return state

def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp7():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.05

    return state

def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp8():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.005

    return state




###
### We do hyperparameter search for (Gaussian/Piecewise) VHRED on Ubuntu:
###
### sdim = 1000
### qdim_encoder = 1000
### qdim_decoder = 2000
### rankdim = 400
### bidirectional_utterance_encoder = True
### reset_utterance_encoder_at_end_of_utterance = False
### reset_utterance_decoder_at_end_of_utterance = True
### lr = 0.0002
### bs = 80
### normop_type = 'LN'

# This is the baseline, with which we compare the 6 experiments below
# Also compare this baseline to prototype_ubuntu_HRED_NormOp_ClusterExp6 (HRED_Exp6 on Guillimin)
def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Baseline_Exp1():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Baseline_Exp2():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp1():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp2():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp3():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp4():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp5():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp6():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    state['patience'] = 20

    return state






###
### We do hyperparameter search for Skip-Thought Vectors and (Gaussian/Piecewise) VHRED on BookCorpus:
###
### qdim_encoder = 1200
### qdim_decoder = 2400
### rankdim = 620
### bidirectional_utterance_encoder = True
### reset_utterance_encoder_at_end_of_utterance = False
### reset_utterance_decoder_at_end_of_utterance = True
### lr = 0.0002
### bs = 80
### normop_type = 'LN'
###
### It should take about 20 days to do one pass over the dataset.
###
### Differences from original Skip-Thought Vector model:
###
### 1. We share parameters between the forward (next utterance) and backward (previous utterance) decoders.
###    The original Skip-Thought Vector model only shared the non-linear transformation applied to
###    the GRU hidden states word embedding output matrix (the GRUs had different parameters).
### 2. In the original Skip-Thought Vector model, examples for the context and both the
###    forward and backward decoder targets were given at the same time. In our model this order
###    is not preserved. Our data iterator only enforces that two adjacent sentences are
###    given to the model in the same batch most of the time.
### 3. In our model, we compute the output probablities in the decoder by applying a linear transformation
###    to both the GRU decoder hidden state and the utterance encoder hidden state. In the original
###    Skip-Thought Vector model, the linear transformation is only applied on the GRU decoder hidden state.
### 4. Optimization hyper-parameters differ. For example, original model clipped gradient norms greater
###    than 10.0, while we clip at 0.01, which is more stable with variational auto-encoders.
### 5. In original Skip-Thought Vector model sentences longer than 30 words were ignored.
###    In our model, sentence pairs longer than 60 words are ignored.
###
### None of these differences should affect performance severely. In fact, I hope they will improve performance.

# These are the three baselines, with which we compare the 6 experiments below
def prototype_book_SkipThought_NormOp():
    state = prototype_state()

    state['end_sym_utterance'] = '</s>'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>


    if 'BOOK_DATA' in os.environ:
        state['train_dialogues'] = os.environ['BOOK_DATA']+'/'+"Train.pairs.pkl"
        state['test_dialogues'] = os.environ['BOOK_DATA']+'/'+"Test.mini.pairs.pkl"
        state['valid_dialogues'] = os.environ['BOOK_DATA']+'/'+"Valid.mini.pairs.pkl"
        state['dictionary'] = os.environ['BOOK_DATA']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../BookCorpus/Train.pairs.pkl"
        state['test_dialogues'] = "../BookCorpus/Test.mini.pairs.pkl"
        state['valid_dialogues'] = "../BookCorpus/Valid.mini.pairs.pkl"
        state['dictionary'] = "../BookCorpus/Dataset.dict.pkl"

    state['save_dir'] = "Output"

    state['valid_freq'] = 5000

    state['prefix'] = "BookModel_SkipThought_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 120
    state['max_grad_steps'] = 62
    state['max_len'] = 60

    state['utterance_decoder_gating'] = 'GRU'
    state['direct_connection_between_encoders_and_decoder'] = True
    state['disable_dialogue_encoder'] = True

    state['qdim_encoder'] = 1200
    state['qdim_decoder'] = 2400
    state['sdim'] = 10
    state['rankdim'] = 620

    state['do_generate_first_utterance'] = False
    state['skip_utterance'] = True
    state['skip_utterance_predict_both'] = True

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 100

    return state


def prototype_book_SkipThought_NormOp_Baseline_Exp1():
    state = prototype_book_SkipThought_NormOp()

    state['qdim_encoder'] = 2400
    state['bidirectional_utterance_encoder'] = False

    return state


def prototype_book_SkipThought_NormOp_Baseline_Exp2():
    state = prototype_book_SkipThought_NormOp()

    return state


def prototype_book_SkipThought_NormOp_Baseline_Exp3():
    state = prototype_book_SkipThought_NormOp()
    state['deep_utterance_decoder_input'] = True

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp1():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp2():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp3():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp4():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp5():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp6():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state

def prototype_reddit_SkipThought_NormOp():
    state = prototype_state()

    state['end_sym_utterance'] = '</s>'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    path = '/home/ml/mnosew1/data/reddit/skip-thought'
    state['train_dialogues'] = "%s/NewsTraining.dialogues.pkl" % path
    state['test_dialogues'] = "%s/NewsTest.dialogues.pkl" % path
    state['valid_dialogues'] = "%s/NewsValidation.dialogues.pkl" % path
    state['dictionary'] = "%s/NewsTraining.dict.pkl" % path

    state['save_dir'] = "Output"

    state['valid_freq'] = 5000

    state['prefix'] = "Reddit_SkipThought_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 120
    state['max_grad_steps'] = 102
    state['max_len'] = 100

    state['utterance_decoder_gating'] = 'GRU'
    state['direct_connection_between_encoders_and_decoder'] = True
    state['disable_dialogue_encoder'] = True

    state['qdim_encoder'] = 1200
    state['qdim_decoder'] = 2400
    state['sdim'] = 10
    state['rankdim'] = 620

    state['do_generate_first_utterance'] = False
    state['skip_utterance'] = True
    state['skip_utterance_predict_both'] = True

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 100

    return state


def prototype_reddit_SkipThought_NormOp_Baseline_Exp1():
    state = prototype_reddit_SkipThought_NormOp()

    return state


def prototype_reddit_SkipThought_NormOp_Baseline_Exp2():
    state = prototype_reddit_SkipThought_NormOp()
    state['deep_utterance_decoder_input'] = True

    return state


def prototype_reddit_SkipThought_NormOp_VAE_Exp1():
    state = prototype_reddit_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_reddit_SkipThought_NormOp_VAE_Exp2():
    state = prototype_reddit_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_reddit_SkipThought_NormOp_VAE_Exp3():
    state = prototype_reddit_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_reddit_SkipThought_NormOp_VAE_Exp4():
    state = prototype_reddit_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state


def prototype_reddit_SkipThought_NormOp_VAE_Exp5():
    state = prototype_reddit_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state


def prototype_reddit_SkipThought_NormOp_VAE_Exp6():
    state = prototype_reddit_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state

def prototype_reddit_HRED_branching_1():
    state = prototype_state()
 
    # Fill your paths here!
    state['train_dialogues'] = "/home/ml/mnosew1/data/reddit/branching/NewsTraining1.dialogues.pkl"
    state['test_dialogues'] = "/home/ml/mnosew1/data/reddit/branching/NewsTesting1.dialogues.pkl"
    state['valid_dialogues'] = "/home/ml/mnosew1/data/reddit/branching/NewsValidation1.dialogues.pkl"
    state['dictionary'] = "/home/ml/mnosew1/data/reddit/branching/NewsTraining1.dict.pkl"
    state['save_dir'] = "Branching1Output"

    state['max_grad_steps'] = 100

    state['valid_freq'] = 5000

    state['prefix'] = "RedditHRED_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 100 # If out of memory, modify this!
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 2000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    return state

def prototype_reddit_HRED_branching_2():
    state = prototype_state()
 
    # Fill your paths here!
    state['train_dialogues'] = "/home/ml/mnosew1/data/reddit/branching/NewsTraining2.dialogues.pkl"
    state['test_dialogues'] = "/home/ml/mnosew1/data/reddit/branching/NewsTesting2.dialogues.pkl"
    state['valid_dialogues'] = "/home/ml/mnosew1/data/reddit/branching/NewsValidation2.dialogues.pkl"
    state['dictionary'] = "/home/ml/mnosew1/data/reddit/branching/NewsTraining1.dict.pkl"
    state['save_dir'] = "Branching2Output"

    state['max_grad_steps'] = 100

    state['valid_freq'] = 5000

    state['prefix'] = "RedditHRED_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 100 # If out of memory, modify this!
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 2000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    return state

def prototype_policy_gradient():
    #state = prototype_state()
    state = {}
    # Fill your paths here!
    #state['train_dialogues'] = "/home/ml/mnosew1/data/twitter/hred/TwitterTrain.dialogues.pkl"
    #state['test_dialogues'] = "/home/ml/mnosew1/data/twitter/hred/TwitterVal.dialogues.pkl"
    #state['valid_dialogues'] = "/home/ml/mnosew1/data/twitter/hred/TwitterTest.dialogues.pkl"
    #state['dictionary'] = "/home/ml/mnosew1/data/twitter/hred/TwitterTrain.dict.pkl"
    
    # This should be the same data as the pretrained model.
    state['train_dialogues'] = "/home/ml/mnosew1/data/twitter/hred_bpe/Train.dialogues.pkl"
    state['test_dialogues'] = "/home/ml/mnosew1/data/twitter/hred_bpe/Valid.dialogues.pkl"
    state['valid_dialogues'] = "/home/ml/mnosew1/data/twitter/hred_bpe/Test.dialogues.pkl"
    state['dictionary'] = "/home/ml/mnosew1/data/twitter/hred_bpe/Dataset.dict.pkl"
    # This should be the filename prefix for the pretrained model.
    state['pretrained_prefix'] = '/home/ml/mnosew1/data/twitter/hred_bpe/SavedModel/1474690048.8_TwitterModel'
    # These parameters will be replaced from the preloaded model.
    state['save_dir'] = "PG_Output"
    state['use_pg'] = True
    state['valid_freq'] = 500
    state['prefix'] = "PG_"
    state['bs'] = 100 # If out of memory, modify this!

    # These parameters should be the same as the pretrained model.
    #state['max_grad_steps'] = 100
    #state['updater'] = 'adam'
    #state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = False
    #state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    #state['direct_connection_between_encoders_and_decoder'] = True
    #state['deep_direct_connection'] = False
    #state['qdim_encoder'] = 200
    #state['qdim_decoder'] = 200
    #state['sdim'] = 100
    #state['rankdim'] = 40
    #state['utterance_decoder_gating'] = 'GRU'

    return state

def prototype_policy_gradient_toy():
    state = prototype_state()
    # Fill your paths here!
    state['train_dialogues'] = "/home/ml/mnosew1/data/synthetic/digits/Training.dialogues.pkl"
    state['test_dialogues'] = "/home/ml/mnosew1/data/synthetic/digits/Val.dialogues.pkl"
    state['valid_dialogues'] = "/home/ml/mnosew1/data/synthetic/digits/Test.dialogues.pkl"
    state['dictionary'] = "/home/ml/mnosew1/data/synthetic/digits/Training.dict.pkl"
    
    # This should be the filename prefix for the pretrained model.
    state['pretrained_prefix'] = None
    # These parameters will be replaced from the preloaded model.
    state['save_dir'] = "PG_Toy_Output"
    state['use_pg'] = True
    state['valid_freq'] = 5000
    state['prefix'] = "PG_"
    state['bs'] = 100 # If out of memory, modify this!

    # These parameters should be the same as the pretrained model.
    state['max_grad_steps'] = 100
    state['updater'] = 'adam'
    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = False
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False
    state['qdim_encoder'] = 100
    state['qdim_decoder'] = 100
    state['sdim'] = 100
    state['rankdim'] = 10
    state['utterance_decoder_gating'] = 'GRU'

    return state
