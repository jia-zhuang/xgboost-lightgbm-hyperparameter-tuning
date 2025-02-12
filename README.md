# XGBoost/LightGBM 调参：基于贝叶斯优化和网格搜索

### XGBoost 需要调节的参数

| 参数 | 取值 | 解释 |
| - | - | - |
| learning_rate | 0 ~ 1 | 学习速率越小，学习得越精细，但所需的学习次数(num_boost_round)越多。默认值 0.1，一般设置 0.05，0.01，0.005 |
| max_depth | > 0 | 树的深度。越大，模型越复杂，表达能力越强，但会过拟合。默认值是 6，常用探索范围 4 ~ 10，根据需要也可以设置更高的树深 |
| min_child_weight | > 0 | 当每个样本权重相等时，min_child_weight 就等于落入每个叶子结点的样本数。可以限制叶子结点继续分割，设置一个大的值可以防止过拟合。默认值是 1，常用探索范围 20 ~ 200，当样本规模很大时，可以设置更大的值 |
| subsample | 0 ~ 1 | 每一轮 boosting 时随机使用一部分样本。可以用来对抗过拟合 |
| colsample_bytree | 0 ~ 1 | 随机使用一部分 feature，可用来对抗过拟合 |

### LightGBM 需要调节的参数

| 参数 | 取值 | 解释 |
| - | - | - |
| learning_rate | 0 ~ 1 | 与 XGBoost 类似 |
| num_leaves | > 0 | 与 XGBoost 中 max_depth 转换关系大致为 num_leaves = 2\*\*max_depth，但树不是完全二叉树，一般设置时会小于 2\*\*max_depth。比如，max_depth 为 7 时效果最好，那么 num_leaves 设为 70 或 80 就够了。默认值为 31 |
| min_data_in_leaf | > 0 | 与 XGBoost 中 min_child_weight 类似。默认值为 20 |
| bagging_fraction | 0 ~ 1 | 与 XGBoost 中 subsample 类似，生效的话必须设置 subsample_freq=1 |
| feature_fraction | 0 ~ 1 | 与 XGBoost 中 colsample_bytree 类似 |
| lambda_l1/l2 | > 0 | L1/L2 正则化参数 |

### [贝叶斯优化](https://github.com/fmfn/BayesianOptimization)

[探索/利用平衡](https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation_vs_exploration.ipynb)

参见代码：bayesHyperTuning.py

### 网格搜索

参见代码：gridHyperTuning.py

