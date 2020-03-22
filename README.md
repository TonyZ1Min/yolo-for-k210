# keras-yolo-for-k210
## 此教程可以完整的在Win完成：制作数据集、训练yolo、转换成k210可用的Kmodel文件
![examlpe](https://github.com/TonyZ1Min/yolo-for-k210/blob/master/example.png)  


-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 

### 1# 下载并安装anaconda3
   Official Website：https://www.anaconda.com/distribution/#download-section
   
   建议从镜像下载：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Windows-x86_64.exe 
   
   *(安装时记得勾选 【Add Anaconda3 to my PATH environment variable】)
   
   
   
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 

### 2# 下载工具 
   下载此工程，在工程根目录下将[train_ann.zip]和[train_img.zip]解压到当前文件夹

   下载ncc工具箱：网盘下载：https://pan.baidu.com/s/1NT2tG4Rv2YJyjOKRh-3t4w  提取码：z9fr
   
   csdn下载：https://download.csdn.net/download/qq_40508193/12261414
   
   将[ncc_0.1_win.zip]放置在工程根目录，解压到当前文件夹
        
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 

### 3# 准备环境 

   Anaconda 命令行，进入工程根目录：
   
   *中国地区建议先给anaconda和pip换源,参照：https://blog.csdn.net/leviopku/article/details/80113021
   
   新建环境： 
   [$ conda create -n yolo python=3.6]
   
   激活环境： 
   [$ conda activate yolo]
   
   安装必要软件包： 
   [(yolo) $ pip install -r requirements.txt]
      
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 


### 4# 修改参数
   在[configs.json]中修改网络类型，lable标签（如raccoon），和其他参数 注意存放图片(train_img)和存放注释(train_ann)的文件夹名称
   ![examlpe](https://github.com/TonyZ1Min/yolo-for-k210/blob/master/cfg.png)
   
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 

 
### 5# 如果只有自己的样本图片(任意尺寸都可以)，还没有VOC格式的xml注释文件，可使用根目录下的[labelImg.exe]进行注释：

    先将图片放在train_img文件夹
    
    Open Dir--->选择存放图片的文件夹(train_img) 
    
    Change Save Dir--->选择存放注释文件夹(train_ann)
    
    Create RectBox--->框选要标注的物体并输入lable，和上文configs中的相同（如raccoon）
    
    Save后点下一个(Next Image)
    会自动生成标注的目标位置的xml文件保存在注释文件夹中
   
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 

 
### 6# 开始训练：
   [(yolo) $ python train.py -c configs.json]   

   等待训练结束，会出现时间命名的文件夹，里面的tflite文件就是训练好的模型,重命名(如：test.tflite)并复制到工程根目录
   
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 

    
### 7# 转换成Kmodel：
   [(yolo) $ ncc_0.1_win\ncc test.tflite test.kmodel -i tflite -o k210model --dataset train_img]

   转换完成根目录会出现test.kmodel，即可烧录进k210中运行
   
-*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- 

   
### 8# 运行
   maixpy程序见(maixpy_code)文件夹，如有修改configs记得修改对应的archor、图像大小(224*224)、lable

## Copyright

* See [LICENSE](LICENSE) for details.
* This project started at [basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras)&[basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras).    
