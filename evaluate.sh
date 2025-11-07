#!/bin/bash

DATASET="csqa"
GRAPH="circle"
ATTACKER_NUM=2
AUDITOR_NUM=4

python3 -m src.mas_consensus.evaluate \
  --model gemini-2.5-flash \
  --dataset $DATASET \
  --graph_types $GRAPH \
  --agent_num 10 \
  --attacker_num 0 > baseline.out

python3 -m src.mas_consensus.evaluate \
  --model gemini-2.5-flash \
  --dataset $DATASET \
  --graph_types $GRAPH \
  --agent_num 12 \
  --attacker_num $ATTACKER_NUM \
  --auditor_num 0 \
  --type 1 > attacker-noauditor.out

python3 -m src.mas_consensus.evaluate \
  --model gemini-2.5-flash \
  --dataset $DATASET \
  --graph_types $GRAPH \
  --agent_num 16 \
  --attacker_num $ATTACKER_NUM \
  --auditor_num $AUDITOR_NUM \
  --type 1 > attacker-auditor.out


tail -n 3 *.out
