model=BidLSTM_E
dataset=SNLI


read -p "would you want to rm *.log, [yes|no]" x
if [ "$x" == "yes" ]; then
  $(rm *.log)
fi

log="train_${dataset}_${model}_$(date +'%m_%d+%H+%M+%S').log"

nohup \
python -u baseline_train.py \
--scratch yes \
--dataset ${dataset} \
--model ${model} \
--epoch 30 \
--batch 128 \
--lr 1e-3 \
--load_model no \
--cuda 3 \
--save_acc_limit 0.79 \
--only_evaluate no \
--skip_loss 0.0 \
>${log} 2>&1 &
