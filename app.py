from flask import Flask, request
import numpy as np
import json
import os
app = Flask(__name__)

from keras.models import load_model
from keras_bert import get_custom_objects
model = load_model('bert/bert_model_0527.h5', custom_objects=get_custom_objects())

pretrained_path  = "bert/"
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

from keras_bert import load_vocabulary
token_dict = load_vocabulary(vocab_path)

from keras_bert import Tokenizer
tokenizer = Tokenizer(token_dict)

@app.route('/getsimilarity',methods=['post'])
def getsimilarity():
    import json
    import regex as re
    result = getResultData()
    data = request.data.decode('utf-8')
    datajson = json.loads(data)
    text = datajson['text']
    id = datajson['id']
    
    try:
        text = re.sub(r'\s+|[，,.。?？]\s*', '', text)
        text_tokenized = test_tokenize(text)
    except Exception as e:
        print(e)
        result['code'] = 501
        result['data']['msg'] = 'Wrong format'
        result['data']['id'] = id
        return json.dumps(result)

    try:
        pred = model.predict(text_tokenized)
    except Exception as e:
        print(e)
        result['code'] = 502
        result['data']['msg'] = 'Cannot predict'
        result['data']['id'] = id
        return json.dumps(result)
    
    result['data']['id'] = id
    if pred[0][0] >= pred[0][1]:
        result['data']['result'] = 0
    elif pred[0][0] < pred[0][1]:
        result['data']['result'] = 1
    result['code'] = 200
    return json.dumps(result)

def getResultData():
    result = {}
    data = {}
    data['id']=''
    data['result']=''
    data['msg']=''
    result['code']=200
    result['data']=data
    return result

def test_tokenize(test_data_x):
    ind = []
    seg = []
    mas = []
    indices,segments = tokenizer.encode(first=test_data_x, max_len=384)
    ind.append(indices)
    seg.append(segments)

    return [np.array(ind),np.array(seg)]
