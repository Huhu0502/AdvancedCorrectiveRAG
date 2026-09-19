## 项目介绍
这是一个学习langgraph以及rag的实践项目，基于之前做的correctiveRAG项目做的增强版本，原项目地址:
https://github.com/Huhu0502/RAGDemo.git
调整旧项目的缺陷，并进行工程化改造。

## 运行环境
python3.11 / Milvus2.5.14 / LangSmith

## 模型配置
* 暂时不考虑模型效果 
* 无论本地模型或者线上模型都可以使用，本项目从赠送免费额度的平台如阿里、智普获取的免费api

在utils/env_utils.py 写死了milvus集合名以及uri
utils/milvus_utils.py可获取连接、建立该集合 以及agent也会默认连接该集合
parser/Milvus_batch_insert.py读取data下的md文件和pdf文件批量插入