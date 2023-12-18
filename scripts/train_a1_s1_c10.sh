python main.py \
--gpu 0 \
--pipeline=multi_teacher \
--teacher resnet8_t \
--student resnet8_t \
--dataset cifar10 --unlabeled cifar10 \
--epochs 500 --fp16 \
--ckpt_path /path/to/saveckpts/ \
--T 10001.0 --alpha 1.0 --seed 1 --w_gan 1 --w_adv 0.2 --use_jsdiv --w_js 5  \
--logfile c10_a1s1_b24_all --use_maxIters \
--from_teacher_ckpt /path/to/teacher_ckpts/ \


