nohup \
python -u baseline_train.py \
--dataset SNLI \
--model LSTM_E \
--epoch 120 \
--batch 512 \
--lr 1e-3 \
--load_model no \
--cuda 1 \
--save_acc_limit 0.80 \
--only_evaluate no \
--skip_loss 0.05 \
>log_snli_train_lstm.log 2>&1 &
