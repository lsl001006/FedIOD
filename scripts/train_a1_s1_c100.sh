
python main.py \
--gpu 0 \
--pipeline=multi_teacher --N_class 100 --z_dim 256 \
--teacher resnet8_t \
--student resnet8_t \
--dataset cifar100 --unlabeled cifar100 \
--epochs 800 --fp16 \
--ckpt_path /path/to/saveckpts/ \
--T 10001.0 --alpha 1 --seed 1 --w_adv 1 \
--logfile a1s1_c100 --use_maxIters --modify_optim_lr \
--from_teacher_ckpt /path/to/teacher_ckpts/ --save_img