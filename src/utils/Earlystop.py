import os


# 每100轮迭代输出状态
if (total_iterations % opt.test_epoch == 0) or (i == totalSteps - 1):



    if Val_acc < best_validation_accuracy:  # 如果当前验证集准确率大于之前的最好准确率
        best_validation_accuracy = Val_acc  # 更新最好准确率
        last_improvement = total_iterations  # 更新上一次提升的迭代轮次
        j_num = 0
        checkpoint_path = os.path.join(self.savePath, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=i)

    else:
        j_num += 1
        if j_num <opt.j_min:
            pass
        else:
            if opt.lr_type == "exponential_decay":
                lr = cur_lr * opt.decay_rate
            elif opt.lr_type == "polynomial_decay":
                global_step = min(i, 1000)
                lr = (cur_lr - 0.00001) * (1 - global_step / 1000) + 0.00001
                '''
                global_step = min(global_step, decay_steps)
                decayed_learning_rate = (learning_rate - end_learning_rate) *
                                        (1 - global_step / decay_steps) ^ (power) +
                                         end_learning_rate
                '''
            # elif opt.lr_type == "natural_exp_decay":
            #     lr = cur_lr * tf.math.exp(-opt.decay_rate * i)
            elif opt.lr_type == "inverse_time_decay":
                lr = cur_lr / (1 + opt.decay_rate * i / 1000)
            else:
                raise ValueError("Your lr_type name is wrong")
            cur_lr = lr