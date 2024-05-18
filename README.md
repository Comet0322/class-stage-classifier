# Class Stage Classifier

# Data
照以下格式放置資料

data/audio/*.wav
data/label/*.txt

# Training
設定 run.sh 裡面的的參數
1. data_dir
2. cache_dir
3. exp_dir
4. splits
5. name
6. model_checkpoint
7. num_epochs
8. lr
9. batch_size
10. segment_length
11. stride
12. grad_accum_steps

訓練模型
```
# 在 terminal 執行以下命令
bash run.sh

# 在 jupyter-notebook 中｀執行以下命令
!bash run.sh
```