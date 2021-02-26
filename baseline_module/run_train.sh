nohup \
python -u baseline_train.py \
--dataset SNLI \
--model Bert_E \
--epoch 120 \
--batch 1024 \
--lr 1e-3 \
--load_model no \
--cuda 3 \
--save_acc_limit 0.73 \
--only_evaluate no \
--skip_loss 0.45 \
>log_snli_train_bert.log 2>&1 &
