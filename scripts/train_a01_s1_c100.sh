python main.py --gpu 0 \
--pipeline=multi_teacher --N_class 100 --z_dim 256 \
--teacher resnet8_t \
--student resnet8_t \
--dataset cifar100 --unlabeled cifar100 \
--epochs 300 --fp16 \
--ckpt_path /path/to/saveckpts/ \
--T 10001.0 --alpha 0.1 --seed 1 --w_gan 0.6 \
--logfile C100_a01_s1_test --use_maxIters \
--from_teacher_ckpt /path/to/teacher_ckpts/ \