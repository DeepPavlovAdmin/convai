#!/usr/bin/env python

from flask import Flask, request, jsonify
from skill import get_results
import uuid
import subprocess

app = Flask(__name__)


@app.route("/respond", methods=['POST'])
def respond():
    text = request.json['text']
    text_filename = '/tmp/text-{}'.format(uuid.uuid4())
    with open(text_filename, 'w') as f:
        print(text, file=f)
    cmd = "./predict_pipeline.sh {}".format(text_filename)

    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    output = str(output, "utf-8").strip().split("\n")[-1]
    result = get_results(output)
    print(result)
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
