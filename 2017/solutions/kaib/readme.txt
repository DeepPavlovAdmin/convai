
Author : Geonmin Kim, Hwaran Lee, CheongAhn Lee, Eunmi Hong, Byeonggeun Lee

Requirement
 - python 3.6
 - ParlAI
 - pytorch
 - tensorflow
 - aiml
  

Chat with kaib bot

step1) place model files in model directory

model/kaib_qa.mdl
model/checkpoint_DA
model/dict_file_th5.dict
model/exp-emb300-hs2048-lr0.0001-bs128


step2) place knowledge base files in kb directory

kb/wikipedia_en_all_nopic_2017-08.zim
kb/index


step3) cd demo

step4) python bot_code/bot.py

step5) chat with @Team_kaib_bot in telegram



Training

1) DA : Dialog act classifier
   directory : DA
   training : python train.py


2) QA
   directory : ParlAI
   setup : python setup.py develop
   training : ./run-squad.sh -e 1 -g 0 -t 1

3) CC
   directory : ParlAI
   setup : python setup.py develop
   training : ./run-dailydialog.sh -e 1 -g 0 -t 1



If you have any questions, feel free to contact us.

