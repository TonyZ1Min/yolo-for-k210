# keras-yolo-for-k210
此教程可以完整的在Win完成：制作数据集、训练yolo、转换成k210可用的Kmodel文件
Train, Convert, Run Yolo on K210 (on Windows)

1# 下载并安装anaconda3 / Download & Steup anaconda3
   Official Website：https://www.anaconda.com/distribution/#download-section
   
   中国地区建议从镜像下载：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Windows-x86_64.exe 
   
   *(安装时记得勾选 【Add Anaconda3 to my PATH environment variable】)
   
   *(remember to choose【Add Anaconda3 to my PATH environment variable】during setting)
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 
2# 下载此工程，在工程根目录下将[train_ann.zip]和[train_img.zip]解压到当前文件夹
   下载ncc工具箱：
   将[ncc_0.1_win.zip]放置在工程根目录，解压到当前文件夹

3# 准备环境 / Prepare the Environment
   Anaconda 命令行，进入工程根目录：
   
   *中国地区建议先给anaconda和pip换源,参照：https://blog.csdn.net/leviopku/article/details/80113021
   
   新建环境 / create a new environment:
   [conda create -n yolo python=3.6]
   
   激活环境 / activate the new environment
   [conda activate yolo]
   
   安装必要软件包 / install necessary packages
   [pip install -r requirements.txt]
   
4# 在[configs.json]中修改网络类型，标签，和其他参数 注意存放图片(train_img)和存放注释(train_ann)的文件夹名称
 
5# 如果只有自己的样本图片，还没有注释文件，使用根目录下的[labelImg.exe]进行注释：
    先将图片放在train_img文件夹
    Open Dir--->选择存放图片的文件夹(train_img) 
    Change Save Dir--->选择存放注释文件夹(train_ann)
    Create RectBox--->框选要标注的物体并输入lable
    Save后点下一个(Next Image)
 
6# 开始训练：[python train.py -c configs.json]   
   等待训练结束，会出现时间命名的文件夹，里面的tflite文件就是训练好的模型,重命名(如：test.tflite)并复制到工程根目录
    
7# 转换成Kmodel：[ncc_0.1_win\ncc test.tflite test.kmodel -i tflite -o k210model --dataset train_img]
   转换完成根目录会出现test.kmodel，即可烧录进k210中运行
   
8# maixpy程序见(maixpy_code)文件夹，如有修改configs记得修改对应的archor、图像大小(224*224)、lable
  
