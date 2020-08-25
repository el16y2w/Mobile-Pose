# cmds = [
#     "CUDA_VISIBLE_DEVICES=0 python train_cmd.py --backbone mobilenetv2 --lr 0.5"
# ]

cmds = [
    # # decay_rate
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_0",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.99 --Model_folder_name A_1",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.90 --Model_folder_name A_2",
    #
    # # # optimizer
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_3",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_4"
    #
    # # #require_improvement
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_5",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.99 --Model_folder_name A_6",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.90 --Model_folder_name A_7",

    #lr_type
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 300 --decay_rate 0.95 --Model_folder_name A_8",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 800 --decay_rate 0.95 --Model_folder_name A_9"
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 300 --decay_rate 0.95 --Model_folder_name A_10",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 800 --decay_rate 0.95 --Model_folder_name A_11"
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 300 --decay_rate 0.95 --Model_folder_name A_12",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Adam --require_improvement 800 --decay_rate 0.95 --Model_folder_name A_13"
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 300 --decay_rate 0.95 --Model_folder_name A_14",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Adam --require_improvement 800 --decay_rate 0.95 --Model_folder_name A_15"
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 300 --decay_rate 0.95 --Model_folder_name A_16",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 800 --decay_rate 0.95 --Model_folder_name A_17"
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Momentum --require_improvement 300 --decay_rate 0.95 --Model_folder_name A_18",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Momentum --require_improvement 800 --decay_rate 0.95 --Model_folder_name A_19"

#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 100 --decay_rate 0.99 --Model_folder_name A_20",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 100 --decay_rate 0.99 --Model_folder_name A_21"
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 100 --decay_rate 0.90 --Model_folder_name A_22",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 100 --decay_rate 0.90 --Model_folder_name A_23"
#
# #require_improvement
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_24",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.99 --Model_folder_name A_25",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.90 --Model_folder_name A_26",
#
#
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_27",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_28"
#
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_29",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_30"
#     # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_31",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_32"
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_33",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Adam --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_34"
#     # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_35",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Adam --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_36"
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_37",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_38"
#     # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_39",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --decay_rate 0.95 --Model_folder_name A_40"
#
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 500 --decay_rate 0.99 --Model_folder_name A_41",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --decay_rate 0.99 --Model_folder_name A_42"
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 500 --decay_rate 0.90 --Model_folder_name A_43",
#     "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --decay_rate 0.90 --Model_folder_name A_44"
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)