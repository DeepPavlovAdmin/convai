import sys
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#sys.path.append("../../DA_cnn") # for main
sys.path.append("../DA_cnn") # for bot.sh

import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

import pdb
import os

class DA_CNN:
    def __init__(self, checkpoint_dir, cuda=False):
        print('initialize DA (CNN) module')

        vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        #pdb.set_trace()
        self.checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        self.graph = tf.Graph()
        self.cuda = cuda

    def classify_user_query(self, query="", passage=""):
        #print('(DA) query = ' + query) # Test PASS
        #print('(DA) passage = ' + passage) # Test PASS
        QA_mode = self.classify_query(query, passage)

        return QA_mode

    def classify_query(self, query, passage):
        with self.graph.as_default():
            #pdb.set_trace()
            if(self.cuda):
                #print('(DA) use cuda')
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
                self.session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
            else:
                #print('(DA) do not use cuda')
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                self.session_conf = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})


            #pdb.set_trace()
            self.sess = tf.Session(config=self.session_conf) # after this, the model is on GPU
            #print('4')
            m_a = []
            with self.sess.as_default():
                # Transform data
                x_test = np.array(list(self.vocab_processor.transform([query])))

                m_data = data_helpers.checkMatch(passage, query)

                if np.size(x_test[0]) < len(m_data):
                     m_temp = m_data[0:np.size(x_test[0])]
                else:
                     m_zero = np.zeros((np.size(x_test[0]) -  len(m_data)), dtype=np.float)
                     m_temp = np.concatenate([m_data, m_zero], 0)

                m_a.append(m_temp)
                m = np.array(m_a)
                
                # Load the saved meta graph and restore variables
                #pdb.set_trace()
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = self.graph.get_operation_by_name("input_x").outputs[0]
                input_m = self.graph.get_operation_by_name("input_m").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                #print('before predictions')
                predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]
                #print('after predictions')

                # Generate batches for one epoch
                #batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
                #x_batches, m_batches = data_helpers.batch_iter2(list(x_test), list(m), 1, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []

                #for x_test_batch, m_test_batch in zip(x_batches, m_batches):
                for x_test_batch, m_test_batch in data_helpers.batch_iter2(list(x_test), list(m), 1, 1, shuffle=False):
                    #print('before sess.run')
                    batch_predictions = self.sess.run(predictions, {input_x: x_test_batch, input_m: m_test_batch, dropout_keep_prob: 1.0})
                    #print('after sess.run')
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

        return int(all_predictions[0])

if __name__ == "__main__":
    #checkpoint_dir = "../model/checkpoint_DA/" # main()
    checkpoint_dir = "/data2/convai_team/1510310818/checkpoints/"
    print(checkpoint_dir)
    da = DA_CNN(checkpoint_dir)

    passage = 'dummy passage'
    # DA case1 : QA
    query = 'where are hesse and bavaria located \?'
    passage = 'the unemployment rate reached its peak of 20 in 2005 since then , it has decreased to 7 in 2013 , which is only slightly above the national average the decrease is caused on the one hand by the emergence of new jobs and on the other by a marked decrease in the working age population , caused by emigration and low birth rates for decades the wages in thuringia are low compared to rich bordering lands like hesse and bavaria therefore , many thuringians are working in other german lands and even in austria and switzerland as weekly commuters nevertheless , the demographic transition in thuringia leads to a lack of workers in some sectors external immigration into thuringia has been encouraged by the government since about 2010 to counter this problem'
    print('label for query = ' + query + ' is ' + str(da.classify_query(query, passage)))

    # DA case2 : CC
    query = 'how are you feeling today ?'
    print('label for query = ' + query + ' is ' + str(da.classify_query(query, passage)))

    # DA case3 : CC
    #pdb.set_trace()
    query = 'hello !'
    print('label for query = ' + query + ' is ' + str(da.classify_query(query, passage)))
 
    passage = "Historically, the name Armenian has come to internationally designate this group of people. It was first used by neighbouring countries of ancient Armenia. The earliest attestations of the exonym Armenia date around the 6th century BC. In his trilingual Behistun Inscription dated to 517 BC, Darius I the Great of Persia refers to Urashtu (in Babylonian) as Armina (in Old Persian; Armina (    ) and Harminuya (in Elamite). In Greek, ??關串館菅恝菅 \"Armenians\" is attested from about the same time, perhaps the earliest reference being a fragment attributed to Hecataeus of Miletus (476 BC). Xenophon, a Greek general serving in some of the Persian expeditions, describes many aspects of Armenian village life and hospitality in around 401 BC. He relates that the people spoke a language that to his ear sounded like the language of the Persians."

    query = 'Historically, the name Armenian has come to internationally designate this group of people. It was first used by neighbouring countries of ancient Armenia. The earliest attestations of the exonym Armenia date around the 6th century BC. In his trilingual Behistun Inscription dated to 517 BC, Darius I the Great of Persia refers to Urashtu (in Babylonian) as Armina (in Old Persian; Armina (    ) and Harminuya (in Elamite). In Greek, ??關串館菅恝菅 \"Armenians\" is attested from about the same time, perhaps the earliest reference being a fragment attributed to Hecataeus of Miletus (476 BC). Xenophon, a Greek general serving in some of the Persian expeditions, describes many aspects of Armenian village life and hospitality in around 401 BC. He relates that the people spoke a language that to his ear sounded like the language of the Persians.'

    print('label for query = ' + query + ' is ' + str(da.classify_query(query, passage)))
