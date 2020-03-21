# keras-yolo-for-k210
此教程可以完整的在Win完成：制作数据集、训练yolo、转换成k210可用的Kmodel文件
Train, Convert, Run Yolo on K210 (on Windows)

1# 下载并安装anaconda3 / Download & Steup anaconda3
   Official Website：https://www.anaconda.com/distribution/#download-section
   
   中国地区建议从镜像下载：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Windows-x86_64.exe 
   
   *(安装时记得勾选 【Add Anaconda3 to my PATH environment variable】)
   
   *(remember to choose【Add Anaconda3 to my PATH environment variable】during setting)
   

2# 准备环境 / Prepare the Environment
   Anaconda 命令行，进入工程根目录：
   
   *中国地区建议先给anaconda和pip换源,参照：https://blog.csdn.net/leviopku/article/details/80113021
   
   新建环境 / create a new environment:
   [conda create -n yolo python=3.6]
   
   激活环境 / activate the new environment
   [conda activate yolo]
   
   安装必要软件包 / install necessary packages
   [pip install -r requirements.txt]
   
