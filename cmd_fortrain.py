cmds = [
    # decay_rate
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_0",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_1",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_2",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_3",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_4",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_5",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_6",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_7",
    # "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_8",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_9",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_10",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_11",

    # optimizer
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_12",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_13",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_14",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_15",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_16",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_17",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_18",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_19",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_20",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_21",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_22",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Adam --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_23",

    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_24",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_25",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_26",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_27",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_28",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_29",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_30",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_31",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_32",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_33",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_34",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Momentum --require_improvement 500 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_35",

    # #require_improvement
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_36",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_37",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_38",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_39",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_40",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_41",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_42",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_43",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_44",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_45",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_46",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_47",

    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_48",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_49",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_50",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_51",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_52",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_53",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_54",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_55",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_56",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_57",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_58",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Adam --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_59",

    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_60",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_61",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_62",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_63",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_64",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_65",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_66",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_67",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_68",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_69",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_70",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Momentum --require_improvement 1000 --test_epoch 50 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_71",

    #Early stopping
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_72",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_73",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_74",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_75",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_76",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_77",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_78",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_79",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type natural_exp_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_80",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.95 --Model_folder_name A_81",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.99 --Model_folder_name A_82",
    "python train_cmd.py --batch 8 --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.001 --optimizer Gradient --require_improvement 500 --test_epoch 20 --j_min 5 --j_max 10 --decay_rate 0.90 --Model_folder_name A_83",


]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)