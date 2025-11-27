#!/bin/bash

MODEL=gemini-2.5-flash
DATASET=adv

mkdir -p evaluation

for file in src/output/${MODEL}/${DATASET}/3/*; do
  file_name=${file##*/}
  python3 -m src.mas_consensus.evaluate --dataset "${DATASET}" --file_path "$file" > "./evaluation/${file_name}"

done

