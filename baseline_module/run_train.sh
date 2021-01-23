nohup \
python -u baseline_train.py \
--dataset IMDB \
--model Bert \
--epoch 10 \
--batch 512 \
--lr 7e-4 \
--load_model no \
--cuda 3 \
--save_acc_limit 0.85 \
>log.log 2>&1 &
