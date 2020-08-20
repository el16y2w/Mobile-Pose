# cmds = [
#     "CUDA_VISIBLE_DEVICES=0 python train_cmd.py --backbone mobilenetv2 --lr 0.5"
# ]

cmds = [
    "python train_cmd.py --batch 5 --backbone mobilenetv2 --lr 0.5 --Model_folder_name 0",
"python train_cmd.py --batch 4 --backbone mobilenetv2 --lr 0.1 --Model_folder_name 1"
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)