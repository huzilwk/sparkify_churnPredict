## 项目环境
```
1.python 3.7
2.pyspark 2.4.4
3.pandas
4.matplotlib
5.seaborn
6.jupyter notebook
```

## 项目动机
使用spark对用户日志数据进行分析，探索预测用户流失的计算模型

## 文件描述
包含 notebook文件：Sparkify-zh-mini.ipynb 和 md文件：README.md

## 结果总结

### 数据探索

##### 付费级别分布

![2级别.png](https://upload-images.jianshu.io/upload_images/13219655-c50dc05fb7925f5b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 人均行为次数小时分布

![6人均使用小时分布.png](https://upload-images.jianshu.io/upload_images/13219655-a67e84046f4fc8df.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### 特征工程

特征计算选取：1.选取性别、2.付费级别、3.注册时间、4.每日在线时长、5.添加好友数、6.收听歌曲数、7.是否升级、8.是否降级、9.点赞数、10.点踩数、11.听歌数、12.听歌手数、13.会话平均歌曲数、14.平均会话数、15.平均会话时长

数据综合之后

```
+-----+-----+
|churn|count|
+-----+-----+
|    1|   60|
|    0|  165|
+-----+-----+
```

### 模型计算结果

*提升树模型：*

混淆矩阵：
 [[ 9  6]
 [ 2 35]]
精准率： 0.6
召回率： 0.8181818181818182
F1得分： 0.6923076923076923

最佳模型参数：
([{'maxIter': 30}, {'maxBins': 40}, {'maxDepth': 2}],avgMetrics: 0.7495034157073681)


