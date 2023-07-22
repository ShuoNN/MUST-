# M2TS: Multi-scale Multi-modal Approach Based on Transformer for Source Code Summarization
We public the source code, datasets and results for the M2TS.
# Datasets
In the M2TS, we use three large-scale datasets for experiments, including two Java and one Python datasets. In data file, we give the three datasets, which obtain from following paper. If you want to train the model, you must download the datasets.
## PYB(Python Barone) dataset
* paper: https://arxiv.org/abs/1707.02275
* data: https://github.com/EdinburghNLP/code-docstring-corpus
## JAH(Java Hu) dataset
* paper: https://xin-xia.github.io/publication/ijcai18.pdf
* data: https://github.com/xing-hu/TL-CodeSum
## JAL(Java Leclair) dataset
* paper: https://arxiv.org/pdf/1904.02660.pdf
* data: http://leclair.tech/data/funcom/
# Data preprocessing
M2TS uses ASTs and source code modalities, which uses the [JDK](http://www.eclipse.org/jdt/) compiler to parse java methods as ASTs, and the [Treelib](https://treelib.readthedocs.io/en/latest/) toolkit to prase Python functions as ASTs. In addition, before embedding ASTs, we use `BERT method` to embed the information of nodes. 
## Get ASTs
In Data-pre file, the `java_get_ast.py` generates ASTs for two Java datasets and `python_get_ast.py` generates ASTs for Python functions. You can run the following command：
将java代码train.code 预处理为train-pre.code ,然后生成ast.json
```
python3 java_get_ast.py source.code ast.json
```
### BERT embedding
Get here to Install the server and client, detail in [here](https://github.com/hanxiao/bert-as-service) 
启动服务器：bert-serving-start  -model_dir /home2/ns/Ns/M2TS-2/M2TS-master/bert/uncased_L-12_H-768_A-12 -num_worker=2 
            打开一个新的终端运行：python server.py
关闭服务器：查看进程：“ netstat -nltp ”，bert服务器地址是0.0.0.5555，“ kill -9 后面的进程号 ”
            查看端口号是5555的进程：netstat -anp | grep 5555
```
pip install bert-serving-server  
pip install bert-serving-client
```
# Train-Test
In Model file, the `_main_.py` enables to train the model. We evaluate the quality of the generated summaries using four evaluation metrics.
Train and test model:  
```
python3 _main_.py
```
The nlg-eval can be set up in the following way, detail in [here](https://github.com/Maluuba/nlg-eval).  
Install Java 1.8.0 (or higher).  
Install the Python dependencies, run:
、、、
没有pip成功,需要手动下载nlg库放入根目录下，然后Python setup.py install,下载文件放入服务器中，……
解决网址：https://blog.csdn.net/qq_36332660/article/details/127880191?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EESLANDING%7Edefault-7-127880191-blog-121844082.pc_relevant_landingrelevant&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EESLANDING%7Edefault-7-127880191-blog-121844082.pc_relevant_landingrelevant&utm_relevant_index=8

pip install git+https://github.com/Maluuba/nlg-eval.git@master

```
# Requirements
If you want to run the model, you will need to install the following packages.  
```
pytorch 1.7.1  
bert-serving-client 1.10.0  
bert-serving-server 1.10.0  
javalang 0.13.0  
nltk 3.5  
networkx 2.5  
scipy 1.1.0  
treelib 1.6.1
```
# Results
In result file, we give the testing results on the datasets. The `Python_pre.txt` is the generated summaries for PYB dataset.
