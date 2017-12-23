#!/bin/bash
exp_dir='exp-dailydialog-scoring'
exp_ptr_dir='exp-dailydialog'
#emb='data/glove.840B.300d.txt'
exp=
gpuid= 
model='scoring_net'
emb=300
hs=1024
lr=0.0001
wd=0 #.00002
attn=false # true / fase
attType=concat  #general concat dot

############### CUSTOM
gradClip=-1

tag='-bs128' #'-bs128'
############### EVALUATION
beam_size=50 #set 0 for greedy search

###############



train=1 # train=1, eval=0
OPTIND=1
while getopts "e:g:t:m:h:b:l:a:w:z:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
		m) model=$OPTARG ;;
		b) embsize=$OPTARG ;;
		h) hs=$OPTARG ;;
		l) lr=$OPTARG ;;
		a) attn=$OPTARG ;;
		w) attn=$OPTARG ;;
		z) tag=$OPTARG;;
	esac
done
shift $((OPTIND -1))

exp=emb${emb}-hs${hs}-lr${lr}
if $attn ; then
	exp=$exp'-a_'${attType}
fi

if [ $(awk 'BEGIN{ print ('$wd' > '0') }') -eq 1 ]; then
	exp=$exp'-wd_'${wd}
fi


exp=${exp}${tag}

### '-' options are defined in parlai/core/params.py 
### -m --model : should match parlai/agents/<model> (model:model_class)
### -mf --model-file : model file name for loading and saving models

if [ $train -eq 1 ]; then # train
	script='examples/train_model_seq2seq_ldecay.py'
	script=${script}' --log-file '$exp_dir'/exp-'${exp}'/exp-'${exp}'.log'
	script=${script}' -bs 128' # training option
	script=${script}' -vparl 34436 -vp 5' #validation option
	script=${script}' -vmt loss -vme -1' #validation measure
	script=${script}' --optimizer adam -lr '${lr}
	
	#Dictionary arguments
	script=${script}' -dbf True --dict-minfreq 5'
fi

if [ $train -eq 0 ]; then # eval
	script='examples/eval_model_human.py'
	script=${script}' --datatype valid'
	script=${script}' --log-file '$exp_dir'/exp-'${exp}'/exp-'${exp}'_eval.log'
	script=${script}' --beam_size '$beam_size
fi

script=${script}' --dict-file exp-dailydialog/dict_file_th5.dict' # built dict (word)

#script=${script}' --embedding_file '$emb #validation option

if [ ! -d ${exp_dir}/exp-${exp} ]; then
	mkdir ${exp_dir}/exp-${exp}
fi

script=${script}' -m '${model}' -t daily_dialog -mf '${exp_dir}/exp-${exp}/exp-${exp} 

if [ -n "$gpuid" ]; then
	script=${script}' --gpu '${gpuid}
fi

python ${script} -hs ${hs} -emb ${emb} -att ${attn} -attType ${attType} -gradClip ${gradClip} -wd ${wd} -ptrmodel ${exp_ptr_dir}/exp-${exp}/exp-${exp}

case "$exp" in
	e300-h2048) python ${script} -hs 1024 -emb 300 -att 0
		;;
esac


