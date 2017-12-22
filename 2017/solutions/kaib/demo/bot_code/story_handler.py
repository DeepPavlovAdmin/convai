import json
import random

class StoriesHandler:
    # Retrieve QA passage
    def __init__(self, filename="data/train-v1.1.json"):
        with open(filename) as f:
            dataset = json.load(f)
        self.stories = [par["context"] for text in dataset["data"] for par in text["paragraphs"]]

    def get_one(self):
        self.current_story = random.choice(self.stories)
        return self.current_story