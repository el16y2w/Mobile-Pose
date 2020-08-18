cmds = [
    "CUDA_VISIBLE_DEVICES=0 python train_cmd.py --backbone mobilenetv2"
]

import os
for cmd in cmds:
    os.system(cmd)