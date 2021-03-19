model=Bert_E
dataset=SNLI


read -p "would you want to rm *.log, [yes|no]" x
if [ "$x" == "yes" ]; then
  $(rm *.log)
fi

log="train_${dataset}_${model}_$(date +'%m_%d+%H+%M+%S').log"

nohup \
python -u baseline_train.py \
--scratch no \
--dataset ${dataset} \
--model ${model} \
--epoch 30 \
--batch 128 \
--lr 9e-4 \
--load_model yes \
--cuda 2 \
--save_acc_limit 0.88 \
--only_evaluate yes \
--skip_loss 0.16 \
>${log} 2>&1 &

tail -f ${log}
