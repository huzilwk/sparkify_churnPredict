## 项目环境
```
-1.pyspark 2.4.4
-2.python 3*
-3.jupyter notebook
-4.pandas
-5.seaborn
```

## 项目动机
使用spark对用户日志数据分析，探索预测用户流失

## 文件描述
包含 notebook文件：Sparkify-zh-mini.ipynb 和 md文件：README.md

## 结果总结

##### 性别分布



![1性别.png](https://upload-images.jianshu.io/upload_images/13219655-6792e44a32db1346.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*性别比例流失与未流失用户相当*

##### 付费级别分布


![2级别.png](https://upload-images.jianshu.io/upload_images/13219655-c50dc05fb7925f5b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*付费用户流失较少*

##### 事件次数和会话时长

![3会话时长.png](https://upload-images.jianshu.io/upload_images/13219655-4ec598224b021d3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*流失用户的会话内事件次数和会话时长都略小于正常用户*

##### 用户行为次数小时分布

![4用户使用小时分布.png](https://upload-images.jianshu.io/upload_images/13219655-20f97a2c20722f57.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



##### 用户行为次数日分布

![5用户使用日分布.png](https://upload-images.jianshu.io/upload_images/13219655-9aa04c62790b2fd5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*用户总体行为次数分布较为相似*

##### 人均行为次数小时分布

![6人均使用小时分布.png](https://upload-images.jianshu.io/upload_images/13219655-a67e84046f4fc8df.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*流失用户的平均行为次数波动较大，更不稳定*

#### 特征工程

特征计算选取：1.选取性别、2.付费级别、3.注册时间、4.每日在线时长、5.添加好友数、6.收听歌曲数、7.是否升级、8.是否降级、9.点赞数、10.点踩数、11.听歌数、12.听歌手数、13.会话平均歌曲数、14.平均会话数、15.平均会话时长

综合15个特征和1个标签构建特征向量，4：1切分为训练集和验证集

数据综合之后

```
+-----+-----+
|churn|count|
+-----+-----+
|    1|   60|
|    0|  165|
+-----+-----+
```

流失用户占比较低，数据标签不平衡，考虑选用能够兼顾精准率和召回率的F1评分来评价模型

### 模型训练

*首先使用逻辑回归、提升树、随机森林三种算法默认值直接计算*

*使用网格搜索和交叉验证的方法调优模型参数*

```
# gbt提升树模型
gbt = GBTClassifier()

paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxIter,[10, 20,30]) \
    .addGrid(gbt.maxBins, [20, 40, 60])\
    .addGrid(gbt.maxDepth,[2, 4, 6, 8,10]) \
    .build()

crossval_gbt = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=3)

Model_gbt = crossval_gbt.fit(train)


gbt_results = Model_gbt.transform(validation)




# 提升树
print('提升树模型：{}'.format(get_evaluate_scores(gbt_results)))

```

输出：

*提升树模型：*

混淆矩阵：
 [[ 9  6]
 [ 2 35]]
精准率： 0.6
召回率： 0.8181818181818182
F1得分： 0.6923076923076923


获取模型参数：

```
#提升树
best_parameters = [(
                [{key.name: paramValue} for key, paramValue in zip(params.keys(),params.values())], metric) \
            for params, metric in zip(
                Model_gbt.getEstimatorParamMaps(),
                Model_gbt.avgMetrics)]
gbt_best_params=sorted(best_parameters,key=lambda el:el[1],reverse=True)[0]
print(gbt_best_params)
```

输出：
([{'maxIter': 30}, {'maxBins': 40}, {'maxDepth': 2}], 0.7495034157073681)

### 总结

分析过程中一度想要将spark数据框转换成pandas数据框直接用sk-learn工具包处理，因为spark在少量数据时的处理效率着实不高，但考虑以后可能的12Gb数据的处理，spark还是首选。

用户流失的定义是非常重要的一步，如果定义成注销和降级用户，占比会高很多，模型处理也会容易很多，但明显不够科学客观反映用户使用现状，所以我综合考虑之后，选择了注销用户和1个月未登录用户，很多用户直接卸载app时未必会注销，1个月沉睡基本可以判定为流失用户。

另外在特征选择上也非常重要，直接影响最后的模型计算结果，尤其是本例中的不平衡标签数据分类。第一次做的时候我选择了10个简单的特征，模型训练的结果不太理想，重新挖掘了5个特征加入之后，模型计算结果取得了一定改善，但还有挖掘空间，例如用户地域数据，使用的浏览器数据等。

最后本次分析使用的数据集是一个mini数据集，目的是建立一个有效的分析模型，能够应用到更大数据集上，（我曾尝试在本地做median量级的数据集，但算力实在有限，放弃了）最终计算结果并不是一个非常令人满意的生产结果，其中一个很重要的原因是数据量太小，不能充分训练模型，所以还无法判断模型在更大数据集上的测试效果，需要继续验证，另外，不平衡分类问题一直是分类器模型的痛点，除了可以在特征工程做进一步的挖掘，模型调优进行更多的尝试，还可以尝试进行过采样和欠采样来平衡样本。
