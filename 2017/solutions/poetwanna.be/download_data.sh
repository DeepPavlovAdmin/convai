#!/bin/bash

set -e

echo -e "\nDownloading the data (about 11G)...\n"
sleep 5
wget https://s3.us-east-2.amazonaws.com/poetwanna.be-data/data.tar.bz2

echo -e "\nExtracting the data (about 30G)...\n"
tar -jxvf data.tar.bz2
rm -f data.tar.bz2

echo -e "\nDONE!\n"
