#!/bin/bash

# This script download data from s3 and puts it into necessary folders

wget https://s3.eu-central-1.amazonaws.com/convai/bot1337-data.tar.gz
tar -zxvf bot1337-data.tar.gz

cp data/model_all_labels.ftz dialog_tracker/data/
cp data/model.t7 question_generation/data/
cp data/textsum_epoch7_14.69_release.t7 opennmt_summarization/models/
cp data/fbchichat_ver2_epoch9.t7 fbnews_chitchat/data/
cp data/glove.6B.100d.txt intent_classifier/data/
cp data/CPU_epoch5_14.62.t7 opennmt_chitchat/data/

# additional data for question answerer
wget https://www.dropbox.com/s/rx7heizq5y8wi2j/factoid_question_answerer_data.tar.gz
tar -zxvf factoid_question_answerer_data.tar.gz
cp -r factoid_question_answerer_data/* factoid_question_answerer
