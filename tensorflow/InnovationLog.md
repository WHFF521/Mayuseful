# 2022.08.08
---
&emsp;&emsp;���ѧϰ���й��������֪ʶ������������ͻ�����������ĥ������ѧϰ�˴�����֪ʶ�������Ҷ���֪�����������
  # һ.anaconda��ʹ��
&emsp;&emsp;��ʵanaconda��ʹ�ò����Ǳ���ģ��������������Ҫ��һ���Զ������õ����⻷���������Ƕ���tensorflow�����кö�汾�Ķ�����
&emsp;&emsp;����˵��anaconda����һ���������⻷���Ĺ��ߣ�������⻷�������Զ���python�ȸ�������İ汾���Ҳ��ܵ��Ա��廷����Ӱ�죬������Ϊ�Ǹ�С�������������Ϊ�ܶ�����ڸ��»��������ڰ汾�Ĳ�ͬ��Ҫ��ͬ�Ļ�������Ȼ�ͻ����д��󣩡�
[�廪Դ����anaconda](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)    
&emsp;&emsp;��װ��ûʲô��˵���ˣ��밲���ĸ��̾Ͱ�װ���ĸ��̣���ס���������˾��С���������Ҫ���Ǵ�Anaconda Prompt(anaconda3)
![[anaconda prompt](anacondaprompt.jpg)](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/anacondaprompt.jpg?raw=true)
�򿪺�ῴ��������������һ��
` (base) C:\Users\WHFF521> `
��~~��ʱ����ֻ��Ҫ������ͨconda���Ϳ�������ʹ����~~��
���ǿ�ʼ�������⻷��
`conda create -n py37ten115 python=3.7`
py37ten115������������ʲô���ֶ��У���������üǵ����֣�python=3.7��ʾ����3.7�汾��python����дҲ�У�����д�ϣ���Ϊ���ã�
ʹ��`conda info -e`���Բ鿴�Ѿ���������Щ�����⻷��
[condainfoe](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/condainfov.jpg?raw=true)
ʹ��`conda activate py37ten115`�������⻷��
![condaactivate](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/condaactivate.jpg?raw=true)
����ǰ�����������Լ�����������ռ������˵����ǰ�ʹ������⻷�����ˡ�
����`conda deactivate`�����˳����⻷��
����tensorflow2.0֮��İ汾ɾ���˺ö�֮ǰ�ĺ��������ºܶ����ᱨ������Ҫ��ִ�еĴ����кܶ�1.0�в��еĺ�����������ѡ������tensorflow-gpu 1.15�汾��gpu�汾Ӧ����ֻ��NVIDIA�Կ������ã�������MX450���Կ���ȫ�����ˡ�
����`conda install --channel https://conda.anaconda.org/hanyucui tensorflow-gpu=1.15`
���ܰ�װ�ˣ���Ҫ���ض�������y�س������ˡ�
���ض���Ҫ�Ǻ����Ͱ�װ�廪Դ���ߵĸ�һ������Դ
![����Դ](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/tuna.jpg?raw=true)
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

���ǿ���ִ��python��俴һ����ǰtensorflow�İ汾
```python
import tensorflow as tf
print(tf.__version__)
 ```
 ����������ִ�ж���python����ʱ��ÿ�仰�������`;\`���ܼ���дȻ�������һ�д���س�һ��ִ��
![pyten115](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/pythontensorflow.jpg?raw=true)

##������Ĳ���д����Ϊج�βŸոտ�ʼ

# ��. CUDA
&emsp;&emsp;���ڲ�ͬ��tensorflow�汾��������Ҫ��ͬ��CUDA�汾��CUDA�������ΪNVIDIA��һЩ���ߣ�tensorflow����Щʵ����Ӳ���Խӿ����ǣ�������ô˵���Լ��ģ��������Լ��ٶ�ѧ��ֱ�����Լ�������װ���ְ汾��CUDAȻ��ȱ����DLL�ļ��ǳ���ĥ���������˱���ܽ���ʦ�Ŀβ�֪�������⻷���������ָ��ֱ��װ������ֱ��װ�Լ�����Ҳ�����ã������Եúܴ����������й�ϵͳ�������ֱ�Ӱ�װC�̾����ˣ�ʡ���鷳��
![cuda](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/CUDA.jpg?raw=true)
����������װ�������汾��CUDA���ǿ��Եģ���ϵͳ���������������汾��CUDA˭��ǰ����˭��
[tensorflow��CUDAƥ��汾��������](https://tensorflow.google.cn/install/source_windows#gpu)

![tencuda](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/tencuda.jpg?raw=true)
&emsp;&emsp;���ݲܽ���ʦ�Ŀγ̽��������³��԰�װ��tensorflow2.1���������˺ܶ�����ҷ���Ҫ�Ȱ�װcudatookit��cudadnnȻ���ٰ�װtensorflow��
`conda install cudatookit=10.1`
` conda install cudnn=7.6`
����conda��װ��cuda�汾���٣��ܶ��Ͼɰ汾ֻ��Ӣΰ��������С�

# 2022.08.10
&emsp;&emsp;���ڸ㶨�ˣ�����Ūһ���ϾɵĻ���ȥƥ����룬����ֱ�ӰѴ���ĳ��°汾֧�ֵĺ������룬������2.0֮��İ汾Ҳ�ǿ�������֮ǰ�汾�Ĵ���ģ�������Ҫ���ĺܶຯ��
## һ��һ��Ҫfreeze�������������
�����Լ��������еĻ���д��requirement.txt���棬Ҳ�������ƻ����ˡ�
![require](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/requirement.jpg?raw=true)
���ڱ���` only integer scalar arrays can be converted to a scalar index`
֮ǰ��һֱ��Ϊ��python�﷨������ʵ����
![error](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/TypeError.jpg?raw=true)

�°汾û��contrib�����⣺
![xavier](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/xavier_initializer.jpg?raw=true)
AttributeError: module 'tensorflow' has no attribute 'get_variable'
��` tf.get_variable`�ĳ�` tf.compat.v1.get_variable`

������������Ĵ��������ร�ɡ�

# PyCharm
&emsp;&emsp;������װ��ϲ����������Ͱ����������ÿһ��������Ŀ���Ͻ�File���settings������Project��������python������Ҫ���Լ��������⻷���Ļ���ֱ���½�conda���л������õ�2����pycharm������һ�е�terminal
![terminal](https://github.com/WHFF521/Mayuseful/blob/main/tensorflow/terminal.jpg?raw=true)
Ȼ��͸���ִ��python�ļ�������Ͱٶ����޲�����Ȼ���û��Ȼ���ˡ�

### ������������Դ
```
�廪��https://pypi.tuna.tsinghua.edu.cn/simple
�����ƣ�http://mirrors.aliyun.com/pypi/simple/
�й��Ƽ���ѧ https://pypi.mirrors.ustc.edu.cn/simple/
��������ѧ��http://pypi.hustunique.com/
ɽ������ѧ��http://pypi.sdutlinux.org/
���꣺http://pypi.douban.com/simple/

```
