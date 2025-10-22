# 翼型设计生成AI系统

基于xfoil标签数据的条件扩散模型，用于根据指定的升力系数(CL)和阻力系数(CD)生成对应的翼型设计。

## 系统概述

本系统实现了"Airplane wing design using generative AI"项目，使用去噪扩散概率模型(DDPM)结合xfoil计算出的空气动力学性能标签，实现自动化翼型设计。

### 主要功能

1. **条件扩散模型训练**: 使用xfoil数据训练条件扩散模型
2. **翼型生成**: 根据指定的CL/CD要求生成对应翼型
3. **性能评估**: 评估生成翼型的质量和准确性
4. **模型验证**: 全面验证模型在不同条件下的表现

## 文件结构

```
├── main.py                           # 主运行脚本
├── XfoilConditionalDataset.py        # xfoil条件数据集
├── XfoilConditionalDiffusion.py     # 条件扩散模型训练
├── XfoilAirfoilGenerator.py         # 翼型生成器
├── AirfoilPerformanceEvaluator.py   # 性能评估器
├── AirfoilDesignValidationSuite.py  # 验证套件
├── xfoil_data.csv                   # xfoil计算结果数据
├── airfoil_images/                  # 翼型图像目录
│   └── contour_proportional/        # 比例化轮廓图像
└── README.md                        # 说明文档
```

## 安装依赖

```bash
pip install torch torchvision
pip install numpy pandas matplotlib
pip install scikit-learn scipy
pip install pillow opencv-python
pip install tqdm seaborn
```

## 使用方法

### 1. 训练模型

```bash
python main.py --mode train
```

这将：
- 加载xfoil数据和翼型图像
- 训练条件扩散模型
- 保存模型到 `xfoil_conditional_model/` 目录

### 2. 生成翼型

#### 单次生成
```bash
python main.py --mode generate --cl 0.8 --cd 0.012 --samples 4
```

#### 交互式生成
```bash
python main.py --mode interactive
```

#### 批量生成
```bash
python main.py --mode batch --conditions-file conditions.csv
```

### 3. 验证模型

```bash
python main.py --mode validate
```

### 4. 评估性能

```bash
python main.py --mode evaluate
```

## 配置说明

### 数据配置
- `xfoil_csv_file`: xfoil计算结果CSV文件
- `image_dir`: 翼型图像目录
- `condition_columns`: 用作条件的性能指标列名
- `alpha_range`: 攻角范围过滤

### 训练配置
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_timesteps`: 扩散时间步数

### 生成配置
- `ddim_steps`: DDIM采样步数
- `num_samples`: 生成样本数

## 数据格式

### xfoil_data.csv格式
```csv
alpha,CL,CD,CDp,CM,Top_Xtr,Bot_Xtr,Airfoil,CL/CD
-5.0,-0.0704,0.00891,0.00295,-0.1022,0.7404,0.1372,airfoil_0001,-7.901234567901236
-4.0,0.0408,0.00862,0.00259,-0.1012,0.6983,0.1671,airfoil_0001,4.733178654292344
...
```

### 批量生成条件文件格式 (CSV)
```csv
CL,CD,CL/CD
0.8,0.012,66.7
1.0,0.015,66.7
1.2,0.018,66.7
```

### 批量生成条件文件格式 (JSON)
```json
[
    {"CL": 0.8, "CD": 0.012, "CL/CD": 66.7},
    {"CL": 1.0, "CD": 0.015, "CL/CD": 66.7},
    {"CL": 1.2, "CD": 0.018, "CL/CD": 66.7}
]
```

## 输出文件

### 训练输出
- `xfoil_conditional_model/final_model.pth`: 最终模型
- `xfoil_conditional_model/model_epoch_*.pth`: 检查点
- `xfoil_conditional_model/condition_scaler_*.npy`: 标准化参数
- `xfoil_conditional_model/training_loss.png`: 训练损失曲线

### 生成输出
- `generated_airfoils/`: 生成的翼型图像
- `xfoil_conditional_model/sample_epoch_*.png`: 训练过程中的样本

### 验证输出
- `validation_results/validation_report.json`: 验证报告
- `validation_results/validation_summary.png`: 验证摘要图
- `validation_results/validation_details.png`: 验证详情图

### 评估输出
- `evaluation/evaluation_report.json`: 评估报告
- `evaluation/condition_comparison.png`: 条件比较图

## 技术细节

### 模型架构
- **基础网络**: U-Net with Residual Blocks
- **条件嵌入**: 多层感知机处理CL/CD等条件
- **时间嵌入**: 正弦位置编码
- **扩散调度**: 余弦调度，1000时间步

### 训练策略
- **优化器**: AdamW with weight decay
- **学习率调度**: ReduceLROnPlateau
- **梯度累积**: 支持大批次训练
- **EMA**: 指数移动平均模型
- **混合精度**: 支持FP16训练

### 采样方法
- **DDIM**: 确定性采样，50步
- **DDPM**: 随机采样，1000步
- **条件引导**: 条件信息引导生成过程

## 性能指标

### 评估指标
- **条件准确性**: R², MAE, 相对误差
- **分布匹配**: KS测试, 统计比较
- **图像质量**: 锐度, 清晰度
- **生成时间**: 单次生成耗时

### 验证测试
- **特定条件测试**: 固定CL/CD值测试
- **条件范围测试**: 随机条件范围测试
- **边界情况测试**: 极值条件测试

## 使用示例

### 示例1: 训练并生成高升力翼型
```bash
# 训练模型
python main.py --mode train

# 生成高升力翼型
python main.py --mode generate --cl 1.2 --cd 0.015 --samples 4
```

### 示例2: 批量生成不同性能的翼型
创建 `test_conditions.csv`:
```csv
CL,CD,CL/CD
0.6,0.008,75.0
0.8,0.012,66.7
1.0,0.015,66.7
1.2,0.018,66.7
```

运行批量生成:
```bash
python main.py --mode batch --conditions-file test_conditions.csv
```

### 示例3: 完整验证流程
```bash
# 训练模型
python main.py --mode train

# 验证模型
python main.py --mode validate

# 评估性能
python main.py --mode evaluate
```

## 注意事项

1. **数据质量**: 确保xfoil数据和图像文件完整且对应
2. **内存使用**: 训练时注意GPU内存使用，可调整batch_size
3. **生成质量**: 不同条件范围的生成质量可能不同
4. **计算时间**: 生成过程需要一定时间，建议使用GPU加速

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 减少grad_accum_steps
   - 使用CPU训练

2. **生成质量差**
   - 增加训练轮数
   - 调整学习率
   - 检查数据质量

3. **条件不匹配**
   - 检查条件列名
   - 验证数据范围
   - 重新标准化

## 扩展功能

### 自定义条件
可以修改 `condition_columns` 配置来使用不同的性能指标作为条件。

### 模型改进
- 添加更多条件信息（如攻角、雷诺数等）
- 使用更先进的扩散模型架构
- 实现条件引导采样

### 评估增强
- 集成xfoil验证
- 添加CFD仿真验证
- 实现多目标优化

## 联系信息

如有问题或建议，请联系jzhongau@connect.ust.hk

---

**注意**: 本系统仅用于研究和教育目的，生成的翼型设计需要经过专业验证才能用于实际应用。
