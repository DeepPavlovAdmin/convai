from flask import Flask, jsonify, request
app = Flask(__name__)

from dialog_encdec import DialogEncoderDecoder
from state import prototype_state
import cPickle
import search

#MODEL_PREFIX = 'Output/1485188791.05_RedditHRED'
MODEL_PREFIX = '/home/ml/mnosew1/SavedModels/Twitter/1489857182.98_TwitterModel'

state_path = '%s_state.pkl' % MODEL_PREFIX
model_path = '%s_model.npz' % MODEL_PREFIX

state = prototype_state()
with open(state_path, 'r') as handle:
    state.update(cPickle.load(handle))

#state['dictionary'] = '/home/ml/mnosew1/data/twitter/hred_bpe/Dataset.dict.pkl'
state['dictionary'] = '/home/ml/mnosew1/SavedModels/Twitter/Dataset.dict-5k.pkl'
print 'Building model...'
model = DialogEncoderDecoder(state)
print 'Building sampler...'
sampler = search.BeamSampler(model)
print 'Loading model...'
model.load(model_path)
print 'Model built.'

HISTORY = []

@app.route('/hred', methods=['POST'])
def hred_response():
    print 'Generating HRED response...'
    text = request.json['result']['resolvedQuery']
    text = text.replace("'", " '")
    context = '<first_speaker> %s </s>' % text.strip().lower()
    HISTORY.append(context)
    print 'History:', HISTORY
    print 'Context:', context
    samples, costs = sampler.sample([' '.join(HISTORY[-4:]),], ignore_unk=True, verbose=False, return_words=True)
    response = samples[0][0].replace('@@ ', '').replace('@@', '')
    HISTORY.append(response)
    response = response.replace('<first_speaker>', '').replace(" '", "'").replace('<at>', '')
    response = response.replace('<second_speaker>', '').strip()
    print 'Response:', response
    response = {'speech': response,
                'displayText': response,
                'source':'HRED'}
    return jsonify(response)
    

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/name', methods=['GET'])
def return_info():
    return jsonify({'first':'mike', 'last':'noseworthy'})

@app.route('/echo', methods=['POST'])
def echo():
    print 'Made it here.'
    print request.json
    #text = request.json['originalRequest']['data']['inputs'][0]['raw_inputs'][0]['query']
    text = request.json['result']['resolvedQuery']
    print text

    response = {'speech': text,
                'displayText': text,
                'data':{},
                'contextOut':[],
                'source':'HRED'}

    return jsonify(response)


if __name__ == '__main__':
    app.run()
