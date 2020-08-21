# cmds = [
#     "CUDA_VISIBLE_DEVICES=0 python train_cmd.py --backbone mobilenetv2 --lr 0.5"
# ]

cmds = [
    # decay_rate
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr 0.001 --optimizer Gradient --require_improvement 100 --decay_rate 0.95 --Model_folder_name A_0",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr 0.001 --optimizer Gradient --require_improvement 100 --decay_rate 0.99 --Model_folder_name A_1",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr 0.001 --optimizer Gradient --require_improvement 100 --decay_rate 0.90 --Model_folder_name A_2",

    # optimizer
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr 0.001 --optimizer Adam --require_improvement 100 --decay_rate 0.95 --Model_folder_name A_3",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr 0.001 --optimizer Momentum --require_improvement 100 --decay_rate 0.95 --Model_folder_name A_4"

    #require_improvement
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr 0.001 --optimizer Gradient --require_improvement 300 --decay_rate 0.95 --Model_folder_name A_5",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr 0.001 --optimizer Gradient --require_improvement 300 --decay_rate 0.99 --Model_folder_name A_6",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr 0.001 --optimizer Gradient --require_improvement 300 --decay_rate 0.90 --Model_folder_name A_7",

    #lr_type
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 300 --decay_rate 0.98 --Model_folder_name A_8",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 300 --decay_rate 0.98 --Model_folder_name A_9"
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 300 --decay_rate 0.98 --Model_folder_name A_10",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type cosine_decay_restarts --lr 0.001 --optimizer Momentum --require_improvement 300 --decay_rate 0.98 --Model_folder_name A_11"

]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)