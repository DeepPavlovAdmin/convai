import argparse
import pyprind
import json
import sys
import os

import inspect

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import features

def main(args):
    # Get list of feature class names
    feature_list = []
    for name, obj in inspect.getmembers(features):
        if inspect.isclass(obj) and name not in ['SentimentIntensityAnalyzer', 'Feature']:
            feature_list.append(name)

    # construct feature file for each data file
    for data_file in args.data:
        print "\nLoading %s..." % data_file
        with open(data_file, 'rb') as handle:
            data = json.load(handle)
        print "got %d examples" % len(data)

        print "building data..."
        # show a progression bar on the screen
        for feat in feature_list:
            print "feature %s" % feat
            bar = pyprind.ProgBar(len(data), monitor=False, stream=sys.stdout)
            data_features = []
            for idx, msg in enumerate(data):
                # create list of Feature instances
                feature_object = features.get(msg['article'], msg['context'], msg['candidate'], [feat])
                feature_object = feature_object[0]  # only one feature at a time
                data_features.append(feature_object.feat.tolist())
                bar.update()
            # saving feature file
            feature_path = data_file.replace('.json', '.features')
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
            with open("%s/%s.json" % (feature_path, feat), "wb") as handle:
                json.dump(data_features, handle)
            print "saved feature file %s/%s.json" % (feature_path, feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs='+', type=str, help="List of files to build features for")
    args = parser.parse_args()
    main(args)
