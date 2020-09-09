from opt import opt
import os
from src.utils.LR import exponential_decay,polynomial_decay,inverse_time_decay


def earlystop(i,j_num,cur_lr,total_iterations,totalSteps,last_improvement,Val_acc,
              best_validation_accuracy,savePath,saver,sess):
    if (total_iterations % opt.test_epoch == 0) or (i == totalSteps - 1):

        if Val_acc < best_validation_accuracy:
            best_validation_accuracy = Val_acc
            last_improvement = total_iterations
            j_num = 0
            checkpoint_path = os.path.join(savePath, 'model')
            saver.save(sess, checkpoint_path, global_step=i)

        else:
            j_num += 1
            if j_num < opt.j_min:
                pass
            else:
                if opt.lr_type == "exponential_decay":
                    lr = exponential_decay(cur_lr)
                elif opt.lr_type == "polynomial_decay":
                    lr = polynomial_decay(i, cur_lr)
                elif opt.lr_type == "inverse_time_decay":
                    lr = inverse_time_decay(i, cur_lr)
                else:
                    raise ValueError("Your lr_type name is wrong")
                cur_lr = lr
        return best_validation_accuracy,last_improvement,j_num
