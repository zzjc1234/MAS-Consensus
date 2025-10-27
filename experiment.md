# Experiment table

A demo for experiment table

## attack_id

Mark agent whose id is in \[0, i-1\] as attacker

## type

1: all attackers maliciously give wrong answer

2: all attackers maliciously give wrong audit, all attackers maliciously give wrong vote (original type 2 and type 3 merged!!!)

CHANGE!!! worker won't vote and auditor won't work. auditor only audit and vote. worker only answer

## num-auditors

| experiments        | datasets | graphs | num-agents | attack_id | type | num-auditors                                                        | reg-turn | sample-id | max-parallel | threads | log-dir                                                                                                                                                                                                                                                                                                |
| ------------------ | -------- | ------ | ---------- | --------- | ---- | ------------------------------------------------------------------- | -------- | --------- | ------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| defense_comparison | all      | all    | 6          | 1         | 1    | 2                                                                   | default  | default   | default      | default | <!-- INFO: This should have original no attack and dense, with attack and no dense and with attack with dense -->                                                                                                                                                                                      |
| defense_comparison | all      | all    | 10         | 1         | 1    | 2                                                                   | default  | default   | default      | default |
| defense_comparison | all      | all    | 10         | 3         | 1    | 2                                                                   | default  | default   | default      | default |
| defense_comparison | all      | all    | 20         | 2         | 1    | 2                                                                   | default  | default   | default      | default |
| defense_comparison | all      | all    | 20         | 5         | 1    | 2                                                                   | default  | default   | default      | default |
| defense_comparison | all      | all    | 40         | 8         | 1    | 2                                                                   | default  | default   | default      | default | <!-- INFO: This only need with attack with dense. Bar plot（这个做有攻击，有防御的即可，实验图2柱状图，做所有数据集，不同节点不同错误数分开，（相同数量的智能体，错误智能体），拓扑结构放在一块 -->                                                                                                    |
| defense_comparison | all      | all    | 6          | 1         | 1，2 | 2 out of 10 wrong（other experiment no wrong or malicious auditor） | default  | default   | default      | default | <!-- INFO: with attack with dense  （这个做有攻击有防御的即可）（实验表格2） -->                                                                                                                                                                                                                       |
| defense_comparison | all      | all    | 6          | 1         | 1    | 2                                                                   | default  | default   | default      | default | <!-- INFO: use different model like GPT3.5 GPT4o, deepseek-v3 and qwen. This should have original no attack and dense, with attack and no dense and with attack with defense 采用不同模型：GPT3.5，GPT4o，deepseek-v3，千问（表格3）这个要做原始无攻击，无防御的。有攻击无防御的和有攻击有防御的） --> |
| defense_comparison | GSM8K    | all    | 6          | 0         | 1    | 2                                                                   | default  | default   | default      | default |
| defense_comparison | GSM8K    | all    | 6          | 1         | 1    | 2                                                                   | default  | default   | default      | default | <!-- INFO: effciency, original system complete task vs. adding audition time (with attack and without attack) （最后做一个统计效率的实验：原系统完成任务的时间，加入审计后完成任务的时间。（无攻击状态与有攻击状态）做单个即可。实验图2）  -->                                                         |

最后一个实验：（6智能体1错误）有攻击无防御的各个中间智能体回答的准确性和加入防御机制以后各个中间智能体回答的准确性。（实验图1（5个拓扑））（画线：不加防御机制所有（6个）智能体准确率，加入防御机制后所有（6个）智能体的准确率）做一个数据集即可。 再做一个多数投票baseline，首先：直接先生成问题答案，然后再判断，直接全体成员参与投票，而不是先部分审计，就是用来和我们方法做对比的不用（无防御，传统方法，我们的方法）。
