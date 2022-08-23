# 2022.08.08
---
&emsp;&emsp;最近学习了有关神经网络的知识，被各种软件和环境的配置折磨。由于学习了大量的知识以至于我都不知道从哪里记起。
  # 一.anaconda的使用
&emsp;&emsp;其实anaconda的使用并不是必须的，但是我们真的需要的一个自定义配置的虚拟环境。尤其是对于tensorflow这种有好多版本的东西。
&emsp;&emsp;简单来说，anaconda就是一个创建虚拟环境的工具，这个虚拟环境可以自定义python等各种软件的版本，且不受电脑本体环境的影响，可以认为是个小的虚拟机。（因为很多软件在更新换代中由于版本的不同需要不同的环境，不然就会运行错误）。
[清华源下载anaconda](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)    
&emsp;&emsp;安装就没什么可说的了，想安在哪个盘就安装在哪个盘，记住安在哪里了就行。我们所需要的是打开Anaconda Prompt(anaconda3)
![anaconda prompt](https://github.com/WHFF521/Mayuseful/blob/main/picture/anacondaprompt.jpg?raw=true)

打开后会看见类似于这样的一行
` (base) C:\Users\WHFF521> `
（~~此时我们只需要精所有通conda语句就可以完美使用了~~）
我们开始创建虚拟环境
`conda create -n py37ten115 python=3.7`
py37ten115是随便起的名字什么名字都行，建议起个好记的名字，python=3.7表示内置3.7版本的python，不写也行（建议写上，因为得用）
使用`conda info -e`可以查看已经创建有哪些的虚拟环境
[condainfoe](https://github.com/WHFF521/Mayuseful/blob/main/picture/condainfov.jpg?raw=true)

使用`conda activate py37ten115`激活虚拟环境
![condaactivate](https://github.com/WHFF521/Mayuseful/blob/main/picture/condaactivate.jpg?raw=true)

看见前面括号里是自己创建的虚拟空间的名字说明当前就处于虚拟环境中了。
输入`conda deactivate`可以退出虚拟环境
由于tensorflow2.0之后的版本删除了好多之前的函数，导致很多代码会报错，我需要的执行的代码有很多1.0中才有的函数，所以我选择下载tensorflow-gpu 1.15版本。gpu版本应该是只有NVIDIA显卡才能用，本机主MX450的显卡完全够用了。
输入`conda install --channel https://conda.anaconda.org/hanyucui tensorflow-gpu=1.15`
就能安装了，需要下载东西输入y回车就行了。
下载东西要是很慢就安装清华源告诉的改一下下载源
![下载源](https://github.com/WHFF521/Mayuseful/blob/main/picture/tuna.jpg?raw=true)
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  ```

我们可以执行python语句看一看当前tensorflow的版本
```python
import tensorflow as tf
print(tf.__version__)
 ```
 在命令行中执行多重python语句的时候每句话后面加上`;\`就能继续写然后在最后一行代码回车一次执行
![pyten115](https://github.com/WHFF521/Mayuseful/blob/main/picture/pythontensorflow.jpg?raw=true)


##我是真的不想写，因为噩梦才刚刚开始

# 二. CUDA
&emsp;&emsp;对于不同的tensorflow版本，我们需要不同的CUDA版本。CUDA可以理解为NVIDIA的一些工具，tensorflow用这些实现软硬件对接可能是（我是这么说服自己的）。当初自己百度学的直接在自己电脑上装各种版本的CUDA然后还缺各种DLL文件非常折磨。后来看了北大曹健老师的课才知道在虚拟环境里可以用指令直接装。对于直接装自己电脑也可以用，但是显得很呆。像这种有关系统的软件就直接安装C盘就行了，省的麻烦。
![cuda](https://github.com/WHFF521/Mayuseful/blob/main/picture/CUDA.jpg?raw=true)
像我这样的装了两个版本的CUDA，是可以的，在系统环境变量中两个版本的CUDA谁在前面用谁。
[tensorflow和CUDA匹配版本官网链接](https://tensorflow.google.cn/install/source_windows#gpu)

![tencuda](https://github.com/WHFF521/Mayuseful/blob/main/picture/tencuda.jpg?raw=true)
&emsp;&emsp;根据曹健老师的课程介绍我重新尝试安装了tensorflow2.1。又遇到了很多错误。我发现要先安装cudatookit和cudadnn然后再安装tensorflow，
`conda install cudatookit=10.1`
` conda install cudnn=7.6`
能用conda安装的cuda版本很少，很多老旧版本只有英伟达官网才有。

# 2022.08.10
&emsp;&emsp;终于搞定了，与其弄一个老旧的环境去匹配代码，不如直接把代码改成新版本支持的函数代码，就算是2.0之后的版本也是可以运行之前版本的代码的，但是需要更改很多函数
## 一定一定要freeze啊，真的是作孽
对于自己代码所有的环境写进requirement.txt里面，也算是行善积德了。
![require](https://github.com/WHFF521/Mayuseful/blob/main/picture/requirement.jpg?raw=true)
对于报错` only integer scalar arrays can be converted to a scalar index`
之前我一直以为是python语法错误，其实不是
![error](https://github.com/WHFF521/Mayuseful/blob/main/picture/TypeError.jpg?raw=true)

新版本没有contrib的问题：
![xavier](https://github.com/WHFF521/Mayuseful/blob/main/picture/xavier_initializer.jpg?raw=true)
AttributeError: module 'tensorflow' has no attribute 'get_variable'
将` tf.get_variable`改成` tf.compat.v1.get_variable`

如果还有其他的错误就自求多福吧。

# PyCharm
&emsp;&emsp;官网安装，喜欢安在哪里就安在哪里，对于每一个工程项目左上角File里打开settings，里面Project里面配置python环境，要用自己建的虚拟环境的话就直接新建conda已有环境。用的2话用pycharm最下面一行的terminal
![terminal](https://github.com/WHFF521/Mayuseful/blob/main/picture/terminal.jpg?raw=true)
然后就各种执行python文件，报错就百度修修补补，然后就没有然后了。

### 附赠国内下载源
```
清华：https://pypi.tuna.tsinghua.edu.cn/simple
阿里云：http://mirrors.aliyun.com/pypi/simple/
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
华中理工大学：http://pypi.hustunique.com/
山东理工大学：http://pypi.sdutlinux.org/
豆瓣：http://pypi.douban.com/simple/

```
