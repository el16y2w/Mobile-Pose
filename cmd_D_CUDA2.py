# cmds = [
#     "CUDA_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python train_cmd.py --backbone mobilenetv2 --lr 0.5"]
from opt import opt
cmds = [

    "CUDA_VISIBLE_DEVICES=2 python train_cmd.py --Early_stopping=False --modeloutputFile 'Yogapose_D/' --checkpoinsaveDir 'Yogapose_D/' --train_all_result 'Result/Yogapose_D/' --batch 8 --backbone mobilenetv2 --lr_type exponential_decay --lr 0.001 --optimizer Gradient --decay_rate 0.95 --Model_folder_name 0",
]

import os
os.makedirs("Result/{}".format(opt.modeloutputFile), exist_ok=True)
if not os.path.exists("Result/Yogapose_D/train_log.log"):
    os.mknod("Result/Yogapose_D/train_log.log")
log = open("Result/Yogapose_D/train_log.log", "a+")
for cmd in cmds:
    log.write("try different dataset")
    log.write(cmd)
    log.write("\n")
    os.system(cmd)