# text-generate
## Char_RNN 生成文本
### 生成过程
1. 首先通过train.py 脚本训练模型，模型参数包括：
```
('name', 'default', 'name of the model')
('num_seqs', 100, 'number of seqs in one batch')
('num_steps', 100, 'length of one seq')
('lstm_size', 128, 'size of hidden state of lstm')
('num_layers', 2, 'number of lstm layers')
('use_embedding',True, 'whether to use embedding')
('embedding_size',64, 'size of embedding')
('learning_rate', 0.001, 'learning_rate')
('is_train',True,"training or predicting")
('input_file', '', 'utf8 encoded text file')
('max_steps', 100000, 'max steps to train')
('save_every_n', 1000, 'save the model every n steps')
('log_every_n', 10, 'log to the screen every n steps')
('max_vocab', 3500, 'max char number')
```
2. 通过sample.py 脚本生成文本，模型参数包括：
```
('lstm_size', 128, 'size of hidden state of lstm')
('num_layers', 2, 'number of lstm layers')
('use_embedding', False, 'whether to use embedding')
('embedding_size', 64, 'size of embedding')
('converter_path', '', 'model/name/converter.pkl')
('checkpoint_path', '', 'checkpoint path')
('start_string', '', 'use this string to start generating')
('max_length', 30, 'max length to generate')
```
### eg：写诗模型
#### 训练
模型参数保存在model/poetry/中
python train.py \
  --use_embedding \
  --input_file ./dataset/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
#### 生成文本
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
#### 结果展示
```
何以江南客，相逢在旧林。
不知江上客，不见此时何。
江湖无限意，江水有春波。
江上春风起，江南水上多。
秋风生旧径，秋月到江村。
南陌春风起，孤城此日长。
江南秋水上，帆尽洞庭秋。
江上春风起，山高海路深。
故人何所见，应是此时同。
水下秋云起，山深海树深。
山中无限事，应是旧相亲。
水外秋云尽，江西水色深。
山山无限路，云色入秋山。
山外人无事，江边月满关。
江湖春草阔，江上水桥深。
不识江南客，应知此别情。
江山有归路，江上有人归。
野寺山边雨，江边月满山。
故乡春草遍，归路向江流。
南北山南路，孤舟一里程。
山中一相送，山路复无人。
南北无时别，孤舟又复赊。
```

