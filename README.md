# MUTS: Source Code Summarization with Multi-Scale Structural Guided Transformer
We public the source code, datasets and results for the MUTS.
# Datasets
In the MUTS, we use two large-scale datasets for experiments, including one Java and one Python datasets. In data file, we give the two datasets, which obtain from following paper. If you want to train the model, you must download the datasets.
## PBD(Python Wan) dataset
* paper: https://dl.acm.org/doi/abs/10.1145/3238147.3238206

## JHD(Java Hu) dataset
* paper: https://xin-xia.github.io/publication/ijcai18.pdf
* data: https://github.com/xing-hu/TL-CodeSum
# Data preprocessing
MUTS uses ASTs and source code modalities, which uses the [JDK](http://www.eclipse.org/jdt/) compiler to parse java methods as ASTs, and the [Treelib](https://treelib.readthedocs.io/en/latest/) toolkit to parse Python functions as ASTs. 
## Get ASTs
In Data-pre file, the `java_get_ast.py` generates ASTs for two Java datasets and `python_get_ast.py` generates ASTs for Python functions. You can run the following command：

```
python java_get_ast.py source.code ast.json
```
# Train-Test
In the Model file, the `RUN.py` enables to train the model. We evaluate the quality of the generated summaries using four evaluation metrics.
Train and test model:  
```
python RUN.py
```
The nlg-eval can be set up in the following way, detail in [here](https://github.com/Maluuba/nlg-eval).  
Install Java 1.8.0 (or higher).  
Install the Python dependencies, run:
、、、
If the pip is not successful, you need to manually download the nlg library into the root directory, and then Python setup.py install, download the file and put it into the server，……
Solution URL：https://blog.csdn.net/qq_36332660/article/details/127880191?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EESLANDING%7Edefault-7-127880191-blog-121844082.pc_relevant_landingrelevant&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EESLANDING%7Edefault-7-127880191-blog-121844082.pc_relevant_landingrelevant&utm_relevant_index=8

pip install git+https://github.com/Maluuba/nlg-eval.git@master

```
# Requirements
If you want to run the model, you will need to install the following packages.  
```
pytorch 1.7.1    
javalang 0.13.0  
nltk 3.5  
networkx 2.5  
scipy 1.1.0  
treelib 1.6.1
```
# Results
In result file, we give the testing results on the datasets. The `Python_pre.txt` is the generated summaries for PBD dataset.
