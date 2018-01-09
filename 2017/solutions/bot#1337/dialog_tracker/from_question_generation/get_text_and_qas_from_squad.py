import json
import subprocess


if __name__ == '__main__':

    filename="data/train-v1.1.json"
    with open(filename) as f:
        dataset = json.load(f)
    stories = [par["context"] for text in dataset["data"] for par in text["paragraphs"]][:25]

    data = []

    for story in stories:
        out = subprocess.check_output(["from_question_generation/get_qnas", story])
        questions = [line.split('\t') for line in str(out, "utf-8").split("\n")]
        factoid_qas = [{'question': e[0], 'answer': e[1], 'score': e[2]} for e in questions if len(e) == 3]
        data.append({'text': story, 'qas': factoid_qas})

    with open('data/squad-50-qas.json', 'w') as f:
        json.dump(data, f)
