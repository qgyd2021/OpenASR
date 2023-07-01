#!/usr/bin/env bash

train_subset=dir/train.json


a=${train_subset##*/}

b=${train_subset%.*}
c=${b##*/}

echo "${a}"
echo "${b}"
echo "${c}"
