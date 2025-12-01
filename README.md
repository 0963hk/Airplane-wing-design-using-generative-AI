# 2025.12.1 周一
建立高质量的几何数据库是训练生成式模型的基础。 本研究选取 UIUC 翼型坐标数据库（UIUC Airfoil Coordinates Database）作为原始数据源，该数据库涵盖了涵盖从低雷诺数模型到超音速翼型的广泛设计空间。为了高效且可靠地构建训练语料库，开发了一套基于 Python 的自动化数据采集管线（Automated Data Acquisition Pipeline）。
该采集模块采用基于会话（Session-based）的 HTTP 请求策略，并通过伪造 User-Agent 标头模拟标准浏览器行为，以确保与服务器交互的稳定性。在解析阶段，算法利用正则表达式（Regular Expressions）从 HTML 源码中精准提取所有 .dat 格式的坐标文件链接，并通过集合运算剔除重复项，构建出唯一的下载索引。为了增强采集过程的鲁棒性并遵守网络伦理，该管线集成了自适应流控机制：通过引入随机化的请求延迟（1-3秒）和乱序下载策略，有效规避了服务器拥塞风险，同时设置了严格的超时处理逻辑以应对网络波动。
另外，为了确保输入数据的物理有效性，我们在采集端实施了实时与后处理双重校验。下载过程中会自动过滤字节数异常（如小于50 bytes）的空文件或损坏文件；采集结束后，通过遍历文件系统的统计分析，进一步验证了数据集的完整性与分布情况。最终，该流程成功构建了一个包含数千个独立翼型文件的本地存储库，为后续的格式清洗与神经网络训练提供了坚实的数据支撑。
The foundation of a robust generative model lies in the diversity and fidelity of its training data. In this study, we designated the UIUC Airfoil Coordinates Database as the primary source, owing to its comprehensive coverage of aerodynamic designs ranging from low-Reynolds-number gliders to supersonic airfoils. To construct the training corpus efficiently, we engineered a custom automated data acquisition pipeline in Python.
The acquisition module utilizes a session-based HTTP request architecture, employing browser-mimicking headers to ensure stable and persistent server interaction. For the indexing phase, the system leverages regular expressions to precisely extract coordinate file links (.dat), using set operations to eliminate duplicates and establish a unique target registry. Crucially, to enhance robustness and adhere to ethical web scraping standards, the pipeline integrates an adaptive flow control mechanism. By implementing randomized request delays (1–3 seconds) and a shuffled download sequence, the system effectively mitigates server load and network timeouts.
To guarantee the physical validity of the dataset, a dual-layer verification protocol was implemented. During the retrieval process, the system performs a real-time integrity check, automatically discarding corrupted artifacts or empty files (e.g., <50 bytes). A post-acquisition validation step further analyzes the file size distribution to confirm data consistency. This rigorous pipeline successfully established a local repository of thousands of distinct airfoil geometries, providing a high-fidelity input stream for the subsequent preprocessing and neural network training stages.

原始数据库中的翼型文件存在显著的数据异构性，主要表现为坐标点排序方向不统一（顺/逆时针混杂）以及部分样本包含非物理的数字化噪声。为了将原始离散点转化为拓扑一致的高质量几何表征，本研究实施了一套严格的预处理算法。
针对坐标排序的多样性，引入了动态遍历机制。通过监测横坐标（x-coordinate）的梯度变化，程序自动识别前缘与后缘的转折点，将分离的上、下翼面坐标重组为连续的单向路径矢量，确保所有样本遵循统一的几何拓扑顺序。在此基础上，通过计算相邻点的欧几里得范数（Euclidean Norm），自动剔除重复点和重叠节点，消除了几何冗余。
同时为了消除数字化过程中引入的高频噪声和锯齿状突变，设计了一种基于二阶差分的曲率一致性滤波器（Curvature-Consistency Filter）。该方法计算坐标点的二阶差分向量及其内积，以近似表征局部曲率变化。利用统计学中的四分位距（Interquartile Range, IQR）法则，算法设定了严格的自适应阈值，精准识别并剔除曲率突变的异常点（Outliers）。这一步骤在保留翼型关键气动特征的同时，有效平滑了局部几何，为后续的图像生成与流体仿真提供了高保真的数据输入。
The raw airfoil files exhibit significant heterogeneity, characterized by inconsistent coordinate ordering (mixed clockwise/counter-clockwise sequences) and non-physical digitization noise. To transform these discrete points into topologically consistent and high-fidelity geometric representations, a rigorous preprocessing algorithm was implemented.
For a start, to address the variability in coordinate sequencing, a dynamic traversal mechanism was introduced. By monitoring the gradient of the x-coordinates, the algorithm automatically identifies the inflection points at the leading and trailing edges. It then reconstructs the separated upper and lower surfaces into a continuous, unidirectional path vector, ensuring a unified geometric topology across all samples. Subsequently, redundant nodes and duplicate points were eliminated by calculating the Euclidean norm of adjacent coordinates, thereby resolving geometric singularities.
After that, to mitigate high-frequency noise and jagged artifacts introduced during digitization, we designed a Curvature-Consistency Filter based on second-order differences. This method approximates local curvature variations by analyzing the inner products of consecutive second-order difference vectors. Leveraging the Interquartile Range (IQR) statistical method, the algorithm applies an aggressive adaptive threshold to surgically detect and remove outliers exhibiting abrupt curvature spikes. This process effectively smooths local geometries while preserving critical aerodynamic features, providing robust data input for the subsequent image generation and fluid dynamics simulations.

# 2025.11.27 周四
- 升阻力系数计算初步解决，但是不确定计算是否准确

# 2025.11.25 周二
- 正在处理xfoil仿真问题
  
# 2025.11.22 周六
- 代码已大部分修复
- 已完成的改进：
  - 特征匹配损失：在生成器损失中加入特征匹配项，使生成图像的特征更接近真实图像，提升连续性。
  - 平滑性损失：惩罚相邻像素差异，鼓励生成更平滑的图像。
  - 改进判别器架构：将判别器模块化，支持提取中间特征用于特征匹配。
  - 学习率调度器：使用余弦退火调度器，逐步降低学习率，提升训练稳定性。
- 已创建 GenerateAirfoil_normalization.py 脚本，功能如下：
  - 标准化归一化：将坐标归一化到 [0, 1] 范围
  - 封闭且光滑：使用插值平滑曲线，确保首尾点连接形成封闭曲线
- 已修改 extract_contour_coordinates，确保坐标点按正确顺序排列：
  - 从后缘点开始（x 坐标最大的点）
  - 沿上翼面到前缘点（x 坐标最小的点）
  - 从前缘点沿下翼面回到后缘点
  - 前缘点只出现一次（在上翼面和下翼面的连接处）
  - 后缘点在开头和结尾出现（形成闭合轮廓）
# 2025.11.18 周二
- 已实现将训练后模型生成的图像像素矩阵转换为光滑连续坐标，可是出现了尾部未封闭且前缘有凹凸的情况
  
# 2025.11.13 周四
- 下一阶段任务：
  - 输出像素矩阵转换为光滑曲线翼型坐标
  - xfoil仿真取最大升阻比翼型

# 2025.11.08 周六
- 翼型坐标数据转换为.npy文件
- 制作ip OUTLINE
- 采用b样条曲线差值使翼型曲线光滑
  
# 2025.11.05 周三
- 更新了原始翼型数据处理方案，能够处理更多类型数据
- 改进了绘图代码
  
# 2025.10.29 周三
- 第一次重阳节放假哈哈哈
- 优化了数据库，翼型图样和气动参数一一对应
- 使用CGAN训练，可是效果并不好，loss曲线很奇怪，明天继续找问题优化
  
# 2025.10.28 周二
- 下周二Design期中，复习提上日程
- 训练机翼的CGAN模型
- eVTOL预计10.31前准备好自己要讲的部分并做PPT
- 拜读Efficient Aerodynamic Shape Optimization with Deep-Learning-Based Geometric Filtering
  
# 2025.10.26 周日
- 学习了Batch Normalization的作用

# 2025.10.25 周六
- DCGAN生成的图像看起来还行呢，不知道为什么生成器和判别器的loss曲线看起来很奇怪，我查了一下说是主要以生成的图像质量为主；
- 组会ppt上会展示从epoch0-epoch200机翼逐渐变‘real’的过程，还蛮有意思；
- DCGAN是CNN和GAN的结合产物，出于对网络的限制，使得生成的结果质量更稳定

# 2025.10.23 周四 
- 降温了；
- 回到GAN的使用上，想着先用DCGAN试试看不带气动参数的翼型生成，只是看看生成的质量咋样；
- 91m晚上怎么要等这么久，app也老出bug...

# 2025.10.22 周三
- 嗯已经有必要写一个项目更新日志的文档了，东西零零碎碎不好整合，还是码字写下来好点；
- 今天忘记把家里电脑开机了，很多文件没法处理，于是在看文献；

## 数据准备
- 翼型轮廓图数据集已基于NACA标准生成，与现有论文不同。
- 原因：网上翼型数据格式不统一，数据清洗会损失大量数据，因此直接生成1000个翼型。
- 生成策略：
  - 70%概率生成4位数翼型及其对称翼型，参数（道高弯度、最大弯度位置、厚度）在合理范围。
  - 30%概率生成5位数翼型，参数（设计升力系数、最大弯度位置、前后弧线类型、厚度）在合理范围。
- 结果：最终获得857个可用翼型文件。

## 翼型轮廓图生成
- 使用翼型数据文件生成了翼型轮廓图。

## 气动特性分析
- 编写Python代码批量调用XFOIL。
- 分析条件：雷诺数Re=1e6、马赫数Ma=0.15、攻角AOA=[-5, 15, 1]。
- 输出数据：cl、cd、cl/cd、cm等，保存为CSV文件。

## 数据探索与可视化
- 对XFOIL气动分析数据进行质量探索：
  - 可视化cl、cd、cl/cd、cm的分布特征。
  - 绘制部分cl随AOA变化曲线和极曲线。
- 主成分分析（PCA）：
  - 对翼型性能特征降维，识别聚类模式。
  - 结果：
    - 密集核心集群：代表性能相似的“主流”翼型。
    - 少数离群点：代表性能独特或极端的“特殊”翼型。

## 模型架构设计
- 主干网络：U-Net（编码器-解码器结构），适合图像生成任务，能捕获多尺度特征。
- 增强机制：
  - 注意力机制：在深层特征图中使用自注意力，聚焦关键几何特征（如前缘、后缘、最大厚度位置）。
  - 条件信息处理：
    - 多层感知机（MLP）编码气动性能参数（CL、CD、CL/CD）。
    - 通过条件调制机制融入每个残差块，约束生成过程。
  - 时间嵌入模块：使用正弦位置编码处理扩散过程的时间步信息。
  - 质量提升：采用EMA（指数移动平均）维护稳定模型副本用于推理。
- 核心思想：模型同时理解翼型几何结构和气动性能关系，通过条件引导生成符合要求的翼型设计。

## 训练过程
- 挑战：训练条件扩散模型具有挑战性，采用以下关键技巧：
  - 数据预处理：
    - 翼型图像归一化到[-1, 1]（扩散模型标准）。
    - 条件参数用StandardScaler标准化，避免量级影响。
  - 训练策略：
    - 余弦退火学习率调度：初始大学习率快速收敛，后期降低避免震荡。
    - 梯度累积：因GPU内存有限，使用小batch size累积梯度，等效增大有效batch size。
  - 损失函数设计：
    - 基础MSE损失。
    - 边缘损失：使用Sobel算子计算图像梯度，提升轮廓清晰度。
  - EMA模型：维护指数移动平均副本，训练用原始模型，生成用EMA模型，提升质量和稳定性。
  - 混合精度训练：使用FP16加速训练并节省显存。
  - 训练周期：200个epoch，每epoch生成样本以监控效果并调整参数。

## 生成过程
- 采样算法：DDIM（比传统DDPM更快）。
- 步骤：
  1. 目标条件标准化（如CL=0.8、CD=0.012），按训练scaler处理。
  2. 从纯噪声图像开始。
  3. DDIM采样：跳过中间步骤，从1000步压缩到50步，加速生成。
    - 每步模型预测噪声，按DDIM更新公式计算下一步图像。
    - 过程类似“去噪”，从噪声逐步生成清晰翼型轮廓。
  4. 条件信息全程参与，确保翼型符合目标性能。
  5. 后处理：图像从[-1, 1]转换到[0, 255]像素值，保存为PNG。
- 结果：生成时间<1秒/翼型，质量高（轮廓清晰、几何合理）。
- 终于也是能成功生成了，回去奖励自己吃点水果:)
- 有点累，早八工作到晚六，晚7上课到晚10...
