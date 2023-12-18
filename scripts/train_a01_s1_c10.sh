python main.py \
--gpu 3 \
--pipeline=multi_teacher \
--teacher resnet8_t \
--student resnet8_t \
--dataset cifar10 --unlabeled cifar10 \
--epochs 500  \
--ckpt_path /path/to/saveckpts/ \
--T 10001.0 --alpha 0.1 --seed 1 --w_gan 1 --w_adv 0 --w_algn 0  --w_baln 0 --w_js 0 --use_jsdiv  \
--logfile C10_a01_s1_onlyGAN_miniter --workers 12 \
--from_teacher_ckpt /path/to/teacher_ckpts/ \
--resume


