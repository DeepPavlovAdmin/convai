#!/usr/bin/env bash

# The directory where the script is
export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export THEANORC=$DIR/config/theano.rc:$HOME/.theanorc

#python modules
export PYTHONPATH=$DIR/libs/pattern:$DIR/libs/lasagne:$DIR/chatbot:$PYTHONPATH
