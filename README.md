
### 项目背景

Sparkify是一款音乐流社交应用，其日志数据记录了用户的使用行为，根据用户历史行为数据判断和预测用户流失对运营有重要意义。本次分析尝试使用Spark ML 对用户行为特征建模，预测用户流失。以下将对一个迷你的子数据集（128MB），是完整数据集（12GB）的一个子集进行探索分析。

### 数据探索

数据概况，由以下数据列构成

```
root
 |-- artist: string (nullable = true)
 |-- auth: string (nullable = true)
 |-- firstName: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- itemInSession: long (nullable = true)
 |-- lastName: string (nullable = true)
 |-- length: double (nullable = true)
 |-- level: string (nullable = true)
 |-- location: string (nullable = true)
 |-- method: string (nullable = true)
 |-- page: string (nullable = true)
 |-- registration: long (nullable = true)
 |-- sessionId: long (nullable = true)
 |-- song: string (nullable = true)
 |-- status: long (nullable = true)
 |-- ts: long (nullable = true)
 |-- userAgent: string (nullable = true)
 |-- userId: string (nullable = true)
```

原数据286500条，清洗userId为空的项后，共278154条，时间跨度2个月

```
+-------+-------------------+
|summary|               time|
+-------+-------------------+
|  count|             278154|
|   mean|               null|
| stddev|               null|
|    min|2018-10-01 08:01:57|
|    max|2018-12-03 09:11:16|
+-------+-------------------+
```


页面种类如下：

```
+-------------------------+
|page                     |
+-------------------------+
|Cancel                   |
|Submit Downgrade         |
|Thumbs Down              |
|Home                     |
|Downgrade                |
|Roll Advert              |
|Logout                   |
|Save Settings            |
|Cancellation Confirmation|
|About                    |
|Settings                 |
|Add to Playlist          |
|Add Friend               |
|NextSong                 |
|Thumbs Up                |
|Help                     |
|Upgrade                  |
|Error                    |
|Submit Upgrade           |
+-------------------------+
```



#### 定义流失用户

流失用户作为预测目标，定义至关重要，page内容中的Cancellation Confirmation项具有明确的指向，注销应视为用户流失，另外挖掘沉睡用户，即一个月未登录用户。

1.注销用户：52
2.沉睡用户：38

合并去重之后：60

####  观察数据分布

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

### 特征工程

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



逻辑回归

```
# 逻辑回归
model_lr = LogisticRegression()
model_lr = model_lr.fit(train)

# 预测
predict_train_lr = model_lr.transform(train)
predict_test_lr = model_lr.transform(validation)
```

混淆矩阵：
 [[ 7  3]
 [ 4 38]]
精准率： 0.7
召回率： 0.6363636363636364
F1得分： 0.6666666666666666



提升树

```
# 提升树
model_gbt = GBTClassifier()
model_gbt = model_gbt.fit(train)

# 预测
predict_train_gbt = model_rf.transform(train)
predict_test_gbt = model_rf.transform(validation)

```

混淆矩阵：
 [[ 5  1]
 [ 6 40]]
精准率： 0.8333333333333334
召回率： 0.45454545454545453
F1得分： 0.5882352941176471



随机森林

```
# 随机森林
model_rf = RandomForestClassifier()
model_rf = model_rf.fit(train)

# 预测
predict_train_rf = model_rf.transform(train)
predict_test_rf = model_rf.transform(validation)

```

混淆矩阵：
 [[ 5  1]
 [ 6 40]]
精准率： 0.8333333333333334
召回率： 0.45454545454545453
F1得分： 0.5882352941176471



未调优情况下，从分类效果来看，F1分数较高的是逻辑回归模型，随机森林和提升树相当。

### 模型调优

*使用网格搜索和交叉验证的方法调优模型参数*

```
# 逻辑回归模型
lr =  LogisticRegression()
paramGrid = ParamGridBuilder() \
    .addGrid(lr.elasticNetParam,[0.0, 0.1, 0.5, 1.0]) \
    .addGrid(lr.regParam,[0.0, 0.05, 0.1]) \
    .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=3)
Model_lr = crossval.fit(train)


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



# 随机森林模型
rf = RandomForestClassifier()

paramGrid = ParamGridBuilder() \
    .addGrid(rf.impurity,['entropy', 'gini']) \
    .addGrid(rf.maxDepth,[2,4,6,8]) \
    .build()

crossval_rf = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=3)

Model_rf = crossval_rf.fit(train)



lr_results = Model_lr.transform(validation)
gbt_results = Model_gbt.transform(validation)
rf_results = Model_rf.transform(validation)


# 逻辑回归
print('逻辑回归模型：{}'.format(get_evaluate_scores(lr_results)))

# 提升树
print('提升树模型：{}'.format(get_evaluate_scores(gbt_results)))

# 随机森林
print('随机森林模型：{}'.format(get_evaluate_scores(rf_results)))
```

输出：
*逻辑回归模型：*

混淆矩阵：
 [[ 4  1]
 [ 7 40]]
精准率： 0.8
召回率： 0.36363636363636365
F1得分： 0.5000000000000001

*提升树模型：*

混淆矩阵：
 [[ 9  6]
 [ 2 35]]
精准率： 0.6
召回率： 0.8181818181818182
F1得分： 0.6923076923076923

*随机森林模型：*


混淆矩阵：
 [[ 6  1]
 [ 5 40]]
精准率： 0.8571428571428571
召回率： 0.5454545454545454
F1得分： 0.6666666666666665

经过调参之后，提升树和随机森林的表现有所提升，逻辑回归反而略有下降，说明BinaryClassificationEvaluator()优化方向未必可以提升F1评分，因为本地运行网格搜索和交叉验证方法太耗时，所以暂不做进一步计算。相比之下，调参之后的提升树分类表现更佳!

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

本次分析使用的数据集是一个mini数据集，目的是建立一个有效的分析模型，能够应用到更大数据集上，最终计算结果并不是一个非常令人满意的生产结果，可以在特征工程做进一步的挖掘，模型调优进行更多的尝试。
