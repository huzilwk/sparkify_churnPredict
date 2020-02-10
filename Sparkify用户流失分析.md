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
模型评估指标选择

```
TP    True Positive     真阳    真实为1，预测也为1
FN    False Negative    假阴    真实为1，预测为0
FP    False Positive    假阳    真实为0，预测为1
TN    True Negative     真阴    真实为0，预测也为0
```
准确率accuracy = （TP + TN）/ （TP+FN+FP+FN）即总样本中预测对的0和1值占比
精准率precision = TP / （TP+FP） 即表示在预测为1的值中真实为1的占比
召回率      recall = TP/  （TP+FN） 即表示在真实为1的值中预测为1的占比
F1评分     F1-score = 2/(1/precision + 1/recall)  即precision和recall的调和平均数

本数据中流失用户即标签1占比28.9%，比例较低，如果选用准确率评价，则全部预测为0，也可达到71.1%的准确率,会误导模型计算； 精准率主要考量标签1的预测准确率，召回率主要考量标签1的查全率，所以考虑选用兼顾二者的F1评分来评价模型，能比较客观反应模型分类能力。

另外介绍下：
敏感度 TPR  = recall
假阳率 FPR =  FP/(FP+TN)  即真实标签0值中预测为1的占比

使用TPR做纵坐标，FPR为横坐标的曲线roc曲线可以衡量二分类模型优劣，fpr越低，tpr越高，模型效果越好，auc即roc曲线下面积可以量化这个指标

![8](/Users/lwk/Documents/data_work/uda/project8/blogPic/8.jpg)



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

The area under ROC for train set is 0.8262014483212629
The area under ROC for test set is 0.8359201773835923
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

The area under ROC for train set is 0.9656023699802498
The area under ROC for test set is 0.8603104212860312

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

The area under ROC for train set is 0.9656023699802498
The area under ROC for test set is 0.8603104212860313

```
混淆矩阵：
 [[ 5  1]
 [ 6 40]]
精准率： 0.8333333333333334
召回率： 0.45454545454545453
F1得分： 0.5882352941176471



未调优情况下，从分类效果来看，F1分数较高的是逻辑回归模型，随机森林和提升树相当，随机森林和提升树模型的训练集auc-roc达到0.96，测试集0.86, 判断过拟合，而逻辑回归的auc_roc值0.82，0.83相近。尝试调优。

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
The area under ROC for train set is: 0.7819289005924942
The area under ROC for test set is: 0.8758314855875834

混淆矩阵：
 [[ 4  1]
 [ 7 40]]
精准率： 0.8
召回率： 0.36363636363636365
F1得分： 0.5000000000000001

*提升树模型：*
The area under ROC for train set is: 0.9671658986175115
The area under ROC for test set is: 0.8325942350332596
混淆矩阵：
 [[ 9  6]
 [ 2 35]]
精准率： 0.6
召回率： 0.8181818181818182
F1得分： 0.6923076923076923

*随机森林模型：*
The area under ROC for train set is: 0.9349078341013822
The area under ROC for test set is: 0.8636363636363639

混淆矩阵：
 [[ 6  1]
 [ 5 40]]
精准率： 0.8571428571428571
召回率： 0.5454545454545454
F1得分： 0.6666666666666665

在花费大量时间进行网格搜索和交叉验证调试之后，未能取得预期的效果，逻辑回归甚至评分降低，提升树和随机森林评分略有提高，但仍然能有较大过拟合可能，训练集和auc-roc值都明显高于测试集，综合来看提升树的评分更高。

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

