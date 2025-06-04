# 🧪 损失函数测试指南

## 📊 可用的损失函数类型

### 1. **binary_cross_entropy** (默认)
标准二分类交叉熵损失函数
```python
# 在config.py中设置
loss_function_type = "binary_cross_entropy"
class_weights = [1.0, 7.0]  # [稳定类权重, 变化类权重]
```

**特点**：
- ✅ 简单直接，训练稳定
- ✅ 通过class_weights处理不平衡数据
- ❌ 不考虑预测置信度和业务成本

---

### 2. **probability_adjusted** 
基础概率调整损失函数
```python
# 在config.py中设置
loss_function_type = "probability_adjusted"
loss_function_params["probability_adjusted"] = {
    "base_stable_prob": 0.95,   # 稳定类基础概率
    "base_change_prob": 0.05,   # 变化类基础概率
}
```

**特点**：
- ✅ 考虑类别的基础分布概率
- ✅ 预测概率超出基础概率时减少惩罚
- ✅ 适合极度不平衡数据
- 🎯 **适用场景**：希望模型不被基础概率误导

---

### 3. **confidence_weighted**
置信度动态权重损失函数
```python
# 在config.py中设置
loss_function_type = "confidence_weighted"
loss_function_params["confidence_weighted"] = {
    "confidence_threshold": 0.8,      # 高置信度阈值
    "high_conf_correct_weight": 0.3,  # 高置信度正确权重
    "high_conf_wrong_weight": 3.0,    # 高置信度错误权重
    "low_conf_weight": 1.0,           # 低置信度权重
}
```

**特点**：
- ✅ 根据预测置信度动态调整惩罚
- ✅ 高置信度错误预测重惩罚
- ✅ 鼓励模型提高预测置信度
- 🎯 **适用场景**：希望模型对自己的预测更"负责"

---

### 4. **business_cost**
业务成本驱动损失函数
```python
# 在config.py中设置
loss_function_type = "business_cost"
loss_function_params["business_cost"] = {
    "false_alarm_cost": 1.0,    # 误报成本(稳定->变化)
    "miss_change_cost": 8.0,    # 漏报成本(变化->稳定)
    "correct_reward": 0.2,      # 正确预测奖励
}
```

**特点**：
- ✅ 直接基于业务错误成本设计
- ✅ 不同类型错误有不同惩罚权重
- ✅ 可以给正确预测奖励
- 🎯 **适用场景**：有明确的业务成本量化

---

### 5. **imbalanced_focal**
改进的Focal Loss
```python
# 在config.py中设置
loss_function_type = "imbalanced_focal"
loss_function_params["imbalanced_focal"] = {
    "alpha": 0.25,              # 类别平衡因子
    "gamma": 2.0,               # 聚焦因子
    "dynamic_adjustment": True,  # 动态调整参数
}
```

**特点**：
- ✅ 自动关注困难样本
- ✅ 内置类别平衡机制
- ✅ 可动态调整参数
- 🎯 **适用场景**：标准方法，广泛验证有效

---

## 🔄 如何切换损失函数

### 方法1：修改配置文件
直接编辑 `experiments/training/config.py`:
```python
# 将这一行改为想要的损失函数类型
loss_function_type: str = "business_cost"  # 例如切换到业务成本损失
```

### 方法2：命令行参数（待实现）
```bash
python main.py --loss-type business_cost --epochs 3
```

---

## 📋 测试建议顺序

### 🥇 第一轮测试：基础对比
1. **binary_cross_entropy** (baseline)
2. **imbalanced_focal** (成熟方案)

### 🥈 第二轮测试：创新方案
3. **confidence_weighted** (置信度驱动)
4. **business_cost** (业务导向)

### 🥉 第三轮测试：概率调整
5. **probability_adjusted** (概率偏差校正)

---

## 📊 评估指标对比

运行每个损失函数后，重点观察：

| 指标 | binary_cross_entropy | imbalanced_focal | confidence_weighted | business_cost | probability_adjusted |
|------|---------------------|------------------|-------------------|---------------|-------------------|
| **稳定类准确率** | ? | ? | ? | ? | ? |
| **变化类准确率** | ? | ? | ? | ? | ? |
| **假阳性数量** | ? | ? | ? | ? | ? |
| **假阴性数量** | ? | ? | ? | ? | ? |
| **训练稳定性** | ? | ? | ? | ? | ? |

---

## 🔧 调参建议

### 如果出现问题：

**训练不收敛**：
- 降低学习率
- 减小损失函数参数的极端值

**过度预测稳定类**：
- 增加变化类的权重/成本
- 调高alpha值（focal loss）

**过度预测变化类**：
- 降低变化类的权重/成本
- 调低alpha值（focal loss）

**训练震荡**：
- 关闭动态调整参数
- 使用更保守的参数设置

---

## 🏃‍♂️ 快速开始

1. **选择损失函数**：在config.py中设置`loss_function_type`
2. **调整参数**：修改对应的`loss_function_params`
3. **运行训练**：`python main.py --epochs 3`
4. **观察结果**：重点看混淆矩阵和类别准确率
5. **记录对比**：建议建立表格记录各方案效果

记住：**每次只改一个变量，便于对比效果！** 