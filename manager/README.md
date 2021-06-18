# manager 使用说明

## 需求

mlflow + hydra

## 用法

- 修改 `configs` 中的绝对路径

- finetune for VE

```python
python run.py +experiment=ve/ft &
```

- finetune for VE (命令行修改默认参数)

```python
python run.py +experiment=ve/ft configs.{CONFIG}=NEW_CONFIG &
```

- soft-prompt for VE

```python
python run.py +experiment=ve/soft-prompt &
```

- soft-prompt for VE (grid search)

```python
python run.py -m +experiment=ve/soft-prompt configs.learning_rate=1.0e-2,3.0e-2,5.0e-2 &
```

- 上述命令的实验结果（模型等）会保存在 `configs.output_dir` 中

- 上述命令的实验记录（metrics、超参数、文件等）会保存在 `mlflow.output_dir` 中的 `mlruns` 文件夹下

- 使用 ui

```
cd [your mlflow.output_dir]
mlflow ui -p PORT
```

本地查看时，连接服务器命令里需要挂载端口

```
ssh -L PORT:127.0.0.1:PORT user@ip
```

之后在本地 `127.0.0.1:PORT` 就可以查看实验记录

- **新建Exp**：参考现有配置，在 `config/experiment/` 中新建一个yaml
