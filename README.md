# SAIL-Pose

T-- Training options
D-- Data options
H-- Hyperparameter options


---------------------------Prepare the environment-------------------------------------------------------------------------------------------------------------------
$pip install -r requirement.txt

In order to check the environment, you can run this command line in terminal:
$python train.py

or run this file in pycharm:
train.py

-----------------------Prepare dataset----------------------------------------------------------------------------------------------------------------------------
Download the dataset you want in Google drive:
For aichallenge: https://drive.google.com/file/d/10BFcnyuQwPY20hp6Bg1yUh8XKDS35qqn/view?usp=sharing
                 https://drive.google.com/file/d/1B3ICSjmapL7AioGOTVz9-eGPrvmoF3wF/view?usp=sharing
For MPIIimages:  https://drive.google.com/file/d/1sBFxUPfrtqnRHp4pFjwDrED9ry7hC7i4/view?usp=sharing
                 https://drive.google.com/file/d/1ukq2VSvKZckd0S4szIJbzBCfjdykb19A/view?usp=sharing


----------------------- Run training code-------------------------------------------------------------------------------------------------------------------------
Before training, please check Config/config_cmd.py for #Train model and 
#data prepare.
Then, check opt.py file, change the option and value you want.

$python train.py





