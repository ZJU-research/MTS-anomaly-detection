2009:
数据集1: 38个数据集（来源于UCR Time Series Data Mining Archive.） --原网址失效，新网站没有搜索到对应数据集
数据集2: 植被覆盖数据（FRAR）+气候数据（降水量、温度、海拔）--未列出明确来源，FRAR数据未搜索到


2017:
数据集1: U.S. Dollar Exchange Rate --网址失效

数据集2: Air Quality Dataset
格式：.csv 长度：9358 维数：15 无标签
文献使用方式：使用了PT08S3 (NOx)、PT08S4 (NO2)、PT08S5 (O3)三个变量，算法为FCM/模糊积分+HMM
数据集来源：https://archive.ics.uci.edu/ml/datasets/Air+Quality
变量名称：


![image](https://user-images.githubusercontent.com/82191552/119921537-c3779a00-bfa0-11eb-89b2-172b3f155a8e.png)

数据集展示：


![image](https://user-images.githubusercontent.com/82191552/119921552-c96d7b00-bfa0-11eb-84d4-8bc9b96b073e.png)





数据集3: EEG Eye State Dataset
格式：.arff（已转换为.csv） 长度：14980 维数：15 无标签
文献使用方式：AF3、F7、FC5, 算法为FCM/模糊积分+HMM
数据集来源：https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
变量名称：前14个变量（AF3-AF4）为脑电图相关数据，eyeDetection为眼睛状态（1为闭眼，0为睁眼）
数据集展示：


![image](https://user-images.githubusercontent.com/82191552/119921561-ce322f00-bfa0-11eb-9ed6-3a331174b70f.png)


2018:
数据集1: Dutch power consumption data --未列出数据来源
数据集2：Intel production dataset 1 --未列出数据来源
数据集3：Intel production dataset 2 --未列出数据来源

数据集4: Etch data（蚀刻数据）：
格式：.mat   有标签
数据集介绍：
包括三次实验、总共129组晶圆的蚀刻数据，其中包括108组正常状态下的数据和21组人为造成的异常状态下的数据。每组数据集长度为100左右，21维
数据集来源：
http://www.eigenvector.com/data/Etch/index.html
变量名称：Time、Step Number、BCl3 Flow、Cl2 Flow、RF Btm Pwr、RF Btm Rfl Pwr、Endpt A、He Press、Pressure、RF Tuner、RF Load、RF Phase Err、RF Pwr、RF Impedance  、TCP Tuner、TCP Phase Err、TCP Impedance、TCP Top Pwr、TCP Rfl Pwr、TCP Load、Vat Valve
文献使用方式：在108组正常序列中混入一组人为制造的异常序列，对这组异常序列进行检测：


![image](https://user-images.githubusercontent.com/82191552/119921576-d5593d00-bfa0-11eb-8096-ecae04ec909d.png)


2021:
数据集1: ECG心电图数据
格式：.dat、.art   有标签
数据集介绍：
	包括48组ECG数据，每组数据包括心电图数据和注释数据。每组中，心电图数据长度为650000，2维。注释数据包括每个R波波峰处的标注信息（同时也仅在R波波峰处有标注信息）。标注信息有42种，包括正常心搏和各类异常心搏等等。
数据集来源：
https://archive.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm
文献使用方式：对包含异常心搏的心电图序列进行检测，计算出序列上每段的异常值：


![image](https://user-images.githubusercontent.com/82191552/119921588-dab68780-bfa0-11eb-80b8-f6171c94a009.png)


第一组0-1000如下图所示，“*”表示R波波峰所在位置：


![image](https://user-images.githubusercontent.com/82191552/119921596-dee2a500-bfa0-11eb-926b-994df8ede447.png)


数据集2: Climate change dataset –网站上暂未找到对应的同名数据集。
vapor (VAP), temperature (TMP), temperature min (TMN), temperature max (TMX), global solar radiation (GLO), extraterrestrial radiation (ETR) and extraterrestrial normal radiation (ETRN)
