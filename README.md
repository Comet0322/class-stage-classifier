# Class Stage Classifier

# Data
照以下格式放置資料

data/audio/*.wav
data/label/*.txt

## 修正標註長度
避免標註總長超出音檔實際長度，執行以下命令統一標註總長為音檔的實際長度。
DATA_DIR 為資料的位置，預設為 "data"
他會將修正好的標註放到到 DATA_DIR 下的 label_fixed
```
python fix_label.py --data_dir DATA_DIR
```
檢查沒問題後將其內容，將舊標註存起來(換個名字)
將 label_fixed 命名為 label

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

# Evaluate
評估 Chat-GPT 預測結果的分數
將模型對測試資料的預測結果放在 label 底下的 pred 資料集 (要自己建立)

```
# 在 terminal 執行以下命令
python3 metric.py
```