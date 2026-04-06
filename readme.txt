2026011649-02素材与源码，本文件夹存放模型训练代码（CCDSP）、web应用代码（数据亦存放在code&web下）、安装说明

①DeepCCDSP.py                     #深度学习CCDSP模型训练代码（会生成一个best_model文件放在code&web中使用）

②在code&web.zib解压后文件夹内部框架及对应功能如下
├── app_flask.py                   # 后端主程序
├── predict_model.py           # 预测模型核心函数
├── templates/                     # HTML模板文件夹
│   ├── index.html                # 首页
│   ├── guide.html               # 系统使用指南
│   ├── reference.html         # 参考数据下载页
│   ├── result.html               # 预测结果可视化页
│   └── predict.html             # 药物预测上传页
└── static/                           # 静态资源文件夹
    ├── css/
    │   ├── style.css          # 通用样式
    │   └── index.css         # 主页专属样式
    └── data/                  # 详解见下

③static/data/下数据介绍
1. GDSC_mutation_input.csv（细胞突变特征）
作用：提供细胞系的基因突变信息
内容：每一行是一个细胞系，每一列是一个基因（如 TP53、KRAS），值表示该基因是否突变
模型用途：进入 突变自编码器，提取突变特征
2. GDSC_SMILE_input.csv（药物分子特征）
作用：提供药物的分子结构 / 指纹特征
内容：每一行是一个药物，用数值向量表示药物化学结构
模型用途：进入 药物自编码器，提取药物特征
3. GDSC_ssgsea_input.csv（细胞通路活性）
作用：提供细胞通路富集分数（ssGSEA）
内容：细胞内各个信号通路的活跃程度（如细胞周期、DNA 修复）
模型用途：不经过自编码器，直接拼接使用，作为细胞功能层面特征

4. GDSC_train_IC50_by_borh_cv00.csv（训练集）
作用：主力训练数据
内容：drug_idx、cell_idx、IC50（药物敏感度实测值）
模型用途：
用 IC50 转换成二分类标签（敏感 = 1，不敏感 = 0）
反向传播更新模型权重
5. GDSC_valid_IC50_by_borh_cv00.csv（验证集）
作用：训练中监控性能、调参、早停
内容：结构同训练集
模型用途：
每轮训练完评估准确率
触发 EarlyStopping 防止过拟合
保存最优模型
6. GDSC_test_IC50_by_borh_cv00.csv（测试集）
作用：最终 unseen 数据评估模型泛化能力
内容：结构同上
模型用途：
输出最终 Accuracy、AUC、混淆矩阵、ROC 曲线
衡量模型真正的预测能力

④安装说明：web应用本地部署说明书