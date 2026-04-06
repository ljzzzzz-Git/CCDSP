import os
import torch #导入PyTorch深度学习框架核心库
import torch.nn as nn
import pandas as pd #读取和处理表格数据
import numpy as np #用于数组和矩阵运算
import matplotlib.pyplot as plt #结果可视化
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report #导入评估指标
from sklearn.preprocessing import StandardScaler #数据标准化工具
from torch.utils.data import Dataset, DataLoader #导入PyTorch数据集和数据加载器工具类
from copy import deepcopy #复制模型参数


#1. 早停机制类
class EarlyStopping:
    #初始化部分
    def __init__(self, patience=5, verbose=True, save_path='best_model.pth'):
        self.patience = patience  #容忍轮数，即验证集性能连续不提升的最大轮数，超则触发早停
        self.verbose = verbose  #是否打印日志信息
        self.counter = 0  #计数器，记录验证集性能连续不提升的轮数
        self.best_score = None  #记录最佳的验证集准确率
        self.early_stop = False  #标志位，标记是否触发早停
        self.save_path = save_path  #保存最优模型的路径

    def __call__(self, val_acc, model):
        score = val_acc
        #情况1：首次运行，直接保存模型
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        #情况2：当前模型没有优于上次训练，计数器++
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} (Best accuracy: {self.best_score:.4f})")
            if self.counter >= self.patience: #计数器到达容忍值，触发早停
                self.early_stop = True
        #当前模型优于上次训练，更新最佳分数，保存当前模型
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    #保存最优模型权重
    def save_checkpoint(self, model):
        if self.verbose:
            print(f"Validation accuracy improved ({self.best_score:.4f} → new best), saving model...")
        torch.save(model.state_dict(), self.save_path)
    #加载最优模型权重
    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))
        return model


#2. 环境与路径配置
DATA_DIR = r"E:\data"
RESULT_DIR = r"E:\result"
os.makedirs(RESULT_DIR, exist_ok=True)

FILES = {
    "drug": os.path.join(DATA_DIR, "GDSC_SMILE_input.csv"), #药物分子特征数据
    "mut": os.path.join(DATA_DIR, "GDSC_mutation_input.csv"), #细胞突变特征数据
    "cell": os.path.join(DATA_DIR, "GDSC_ssgsea_input.csv"), #细胞同路特征数据
    "train": os.path.join(DATA_DIR, "GDSC_train_IC50_by_borh_cv00.csv"), #训练集
    "valid": os.path.join(DATA_DIR, "GDSC_valid_IC50_by_borh_cv00.csv"), #验证集
    "test": os.path.join(DATA_DIR, "GDSC_test_IC50_by_borh_cv00.csv") #测试集
}

ARGS = {
    "code_dim": 64,#自编码器编码后的特征维度
    "drug_hidden_dims": [256, 128],#药物自编码器隐藏层
    "mut_hidden_dims": [256, 128],#突变自编码器隐藏层
    "forward_hidden1": 128,#前馈网络隐藏层1
    "forward_hidden2": 64,#2
    "batch_size": 32,#批量大小
    "epochs": 100,#最大训练轮数
    "lr": 1e-4,#学习率
    "dropout_rate": 0.3,#Dropout率，防止过拟合
    "threshold": 0.5,#分类阈值
    "early_stop_patience": 5,#早停容忍轮数
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


#3. 数据预处理
def load_data(file_paths): #加载所有文件，返回数据字典
    data = {}
    for key, path in file_paths.items():
        if not os.path.exists(path): #检查路径
            raise FileNotFoundError(f"File not found: {path}")
        data[key] = pd.read_csv(path, sep=',', header=0)
        print(f"Loaded {key} data: shape {data[key].shape}, first 3 columns: {list(data[key].columns[:3])}")
    return data


def ic50_to_binary(raw_data, threshold_type="median"): #将IC50值转化为二分类标签，敏感=1，不敏感=0
    for key in ["train", "valid", "test"]:
        if "IC50" not in raw_data[key].columns:
            raise KeyError(f"{key} dataset missing 'IC50' column, check data format")
    #合并所有数据的IC50值，计算阈值
    all_ic50 = pd.concat([
        raw_data["train"]["IC50"],
        raw_data["valid"]["IC50"],
        raw_data["test"]["IC50"]
    ])
    if threshold_type == "median":
        binary_threshold = all_ic50.median()
    else:
        binary_threshold = 10
    print(
        f"\nIC50 binary threshold: {binary_threshold:.4f} (IC50 < threshold = sensitive (1), else = non-sensitive (0))")
    #对三个数据集生成二分类标签
    for key in ["train", "valid", "test"]:
        raw_data[key]["sensitivity_label"] = (raw_data[key]["IC50"] < binary_threshold).astype(int)

    return raw_data, binary_threshold

#进行特征标准化，（x-均值）/标准差，这里只使用了训练集
def standardize_features(train_feat, valid_feat, test_feat):
    scaler = StandardScaler() #初始化标准化器
    train_feat_std = scaler.fit_transform(train_feat) #训练集，拟合+转换
    valid_feat_std = scaler.transform(valid_feat) #验证集，转换
    test_feat_std = scaler.transform(test_feat) #测试集，转换
    return train_feat_std, valid_feat_std, test_feat_std, scaler

#提取药物，突变，细胞特征，进行标准化
def get_standardized_features(processed_data):
    for key in ["drug", "mut", "cell"]:
        if "drug_idx" not in processed_data[key].columns and key == "drug":
            raise KeyError(f"{key} dataset missing 'drug_idx' column")
        if "cell_idx" not in processed_data[key].columns and key in ["mut", "cell"]:
            raise KeyError(f"{key} dataset missing 'cell_idx' column")
    #提取药物特征
    drug_feat = processed_data["drug"].iloc[:, 2:].values
    drug_idx_map = dict(zip(processed_data["drug"]["drug_idx"], drug_feat))
    #提取突变特征
    mut_feat = processed_data["mut"].iloc[:, 2:].values
    mut_idx_map = dict(zip(processed_data["mut"]["cell_idx"], mut_feat))
    #提取细胞通路特征
    cell_feat = processed_data["cell"].iloc[:, 2:].values
    cell_idx_map = dict(zip(processed_data["cell"]["cell_idx"], cell_feat))

    standardized = {} #用于储存标准化后的所有数据
    #处理验证集
    train_df = processed_data["train"]
    train_drug = np.array([drug_idx_map[idx] for idx in train_df["drug_idx"]])
    train_mut = np.array([mut_idx_map[idx] for idx in train_df["cell_idx"]])
    train_cell = np.array([cell_idx_map[idx] for idx in train_df["cell_idx"]])
    #分别对三类特征进行标准化
    train_drug_std, _, _, drug_scaler = standardize_features(train_drug, train_drug, train_drug)
    train_mut_std, _, _, mut_scaler = standardize_features(train_mut, train_mut, train_mut)
    train_cell_std, _, _, cell_scaler = standardize_features(train_cell, train_cell, train_cell)
    standardized["train"] = (train_drug_std, train_mut_std, train_cell_std, train_df["sensitivity_label"].values) #保存标准化数据
    #处理验证集
    valid_df = processed_data["valid"]
    valid_drug = np.array([drug_idx_map[idx] for idx in valid_df["drug_idx"]])
    valid_mut = np.array([mut_idx_map[idx] for idx in valid_df["cell_idx"]])
    valid_cell = np.array([cell_idx_map[idx] for idx in valid_df["cell_idx"]])
    #用训练集的标准化器转换验证集
    valid_drug_std = drug_scaler.transform(valid_drug)
    valid_mut_std = mut_scaler.transform(valid_mut)
    valid_cell_std = cell_scaler.transform(valid_cell)
    standardized["valid"] = (valid_drug_std, valid_mut_std, valid_cell_std, valid_df["sensitivity_label"].values)
    #处理测试集
    test_df = processed_data["test"]
    test_drug = np.array([drug_idx_map[idx] for idx in test_df["drug_idx"]])
    test_mut = np.array([mut_idx_map[idx] for idx in test_df["cell_idx"]])
    test_cell = np.array([cell_idx_map[idx] for idx in test_df["cell_idx"]])
    #使用训练集的标准化器处理测试集
    test_drug_std = drug_scaler.transform(test_drug)
    test_mut_std = mut_scaler.transform(test_mut)
    test_cell_std = cell_scaler.transform(test_cell)
    standardized["test"] = (test_drug_std, test_mut_std, test_cell_std, test_df["sensitivity_label"].values)
    #保存所有标准化器，用于后续使用
    scaler_dict = {"drug": drug_scaler, "mut": mut_scaler, "cell": cell_scaler}
    np.save(os.path.join(RESULT_DIR, "scaler_dict.npy"), scaler_dict)

    return standardized, scaler_dict


#4. 模型定义
class DeepAutoencoderThreeHiddenLayers(nn.Module): #深度自编码器，用于特征的降维和提取
    #初始化函数，输入维度，隐藏层维度，编码维度，激活函数，是否使用dropout
    def __init__(self, input_dim, hidden_dims, code_dim, activation_func=nn.ReLU,
                 code_activation=True, dropout=False, dropout_rate=0.5):
        super().__init__()
        #构建编码器，将高维特征压缩为低维编码
        encoder_modules = []
        #第一层：输入层到第一个隐藏层
        encoder_modules.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_modules.append(activation_func())
        if dropout:
            encoder_modules.append(nn.Dropout(dropout_rate))
            #构建后续隐藏层
        for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:]):
            encoder_modules.append(nn.Linear(in_dim, out_dim))
            encoder_modules.append(activation_func())
            if dropout:
                encoder_modules.append(nn.Dropout(dropout_rate))
        #最后一层：最后一个隐藏层到编码层
        encoder_modules.append(nn.Linear(hidden_dims[-1], code_dim))
        if code_activation:
            encoder_modules.append(activation_func())
            #编码层组合为序列模型
        self.encoder = nn.Sequential(*encoder_modules)
        #解码器：将低维编码还原为原始特征
        decoder_modules = []
        #第一层：编码层到最后一个隐藏层
        decoder_modules.append(nn.Linear(code_dim, hidden_dims[-1]))
        decoder_modules.append(activation_func())
        if dropout:
            decoder_modules.append(nn.Dropout(dropout_rate))
        #构建反向隐藏层
        for in_dim, out_dim in zip(hidden_dims[::-1], hidden_dims[-2::-1]):
            decoder_modules.append(nn.Linear(in_dim, out_dim))
            decoder_modules.append(activation_func())
            if dropout:
                decoder_modules.append(nn.Dropout(dropout_rate))
        #最后一层：第一个隐藏层到输入层
        decoder_modules.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_modules.append(nn.Sigmoid())
        #解码器层组合为序列模型
        self.decoder = nn.Sequential(*decoder_modules)
    #前向传播函数
    def forward(self, x):
        code = self.encoder(x) #编码：提取低维核心特征
        recon = self.decoder(code) #解码：冲返原始特征
        return code, recon

#两层前馈神经网络：用于最终的二分类预测
class ForwardNetworkTwoHiddenLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, activation_func=nn.ReLU):
        super().__init__()
        #构建分类网络：全连接层+批归一化+激活函数+dropout
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), #输入层->第一隐藏层
            nn.BatchNorm1d(hidden_dim1), #批归一化，加速训练
            activation_func(), #激活函数
            nn.Dropout(ARGS["dropout_rate"]), #dropout
            nn.Linear(hidden_dim1, hidden_dim2), #第一隐藏层->第二隐藏层
            nn.BatchNorm1d(hidden_dim2),
            activation_func(),
            nn.Dropout(ARGS["dropout_rate"]),
            nn.Linear(hidden_dim2, 1), #第二隐藏层->输出层
            nn.Sigmoid() #输出0~1之间的概率
        )

    def forward(self, x):
        return self.layers(x) #前向传播，输出分类概率

#主模型，融合药物自编码器，突变自编码器，分类网络
class DEERS_Concat(nn.Module):
    def __init__(self, drug_autoencoder, mut_autoencoder, forward_network):
        super().__init__()
        self.drug_autoencoder = drug_autoencoder
        self.mut_autoencoder = mut_autoencoder
        self.forward_network = forward_network
    #前向传播，输入三类特征，输出概率预测和重构特征
    def forward(self, drug_feat, mut_feat, cell_feat):
        drug_code, drug_recon = self.drug_autoencoder(drug_feat)
        mut_code, mut_recon = self.mut_autoencoder(mut_feat)
        concat_feat = torch.cat((drug_code, mut_code, cell_feat), dim=1) #拼接药物编码，突变编码，原始细胞特征
        pred_prob = self.forward_network(concat_feat) #输入分类网络，输出预测概率
        return pred_prob, drug_recon, mut_recon

#自定义混合损失函数：分类损失+药物重构损失+突变重构损失
class MergedBCELoss(nn.Module):
    def __init__(self, pred_weight=1.0, drug_recon_weight=0.1, mut_recon_weight=0.2):
        super().__init__()
        self.pred_weight = pred_weight #分类损失权重
        self.drug_recon_weight = drug_recon_weight #药物重构损失权重
        self.mut_recon_weight = mut_recon_weight #突变损失重构函数
        self.pred_criterion = nn.BCELoss() #二分类交叉熵损失
        self.recon_criterion = nn.MSELoss() #均方误差损失
    #前向传播：计算总损失
    def forward(self, pred_prob, drug_recon, mut_recon, drug_feat, mut_feat, true_label):
        # 分类损失：预测概率 vs 真实标签
        pred_loss = self.pred_criterion(pred_prob.squeeze(), true_label.float())
        # 药物重构损失：重构特征 vs 原始特征
        drug_recon_loss = self.recon_criterion(drug_recon, drug_feat)
        # 突变重构损失：重构特征 vs 原始特征
        mut_recon_loss = self.recon_criterion(mut_recon, mut_feat)
        #计算损失总和
        total_loss = (self.pred_weight * pred_loss) + \
                     (self.drug_recon_weight * drug_recon_loss) + \
                     (self.mut_recon_weight * mut_recon_loss)
        return total_loss, pred_loss, drug_recon_loss, mut_recon_loss


#5. 数据集与DataLoader
class DrugSensitivityDataset(Dataset): # 自定义数据集类：继承PyTorch Dataset，用于封装训练数据
    def __init__(self, drug_feat, mut_feat, cell_feat, labels):
        # 将numpy数组转换为PyTorch浮点型张量
        self.drug_feat = torch.tensor(drug_feat, dtype=torch.float32)
        self.mut_feat = torch.tensor(mut_feat, dtype=torch.float32)
        self.cell_feat = torch.tensor(cell_feat, dtype=torch.float32)
        # 标签转换为长整型张量
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): # 返回数据集样本总数
        return len(self.labels)

    def __getitem__(self, idx): # 根据索引获取单个样本
        return (
            self.drug_feat[idx],
            self.mut_feat[idx],
            self.cell_feat[idx],
            self.labels[idx]
        )

# 创建训练/验证/测试数据加载器
def create_dataloaders(standardized_data):
    loaders = {}
    num_workers = 4 if ARGS["device"].type == "cuda" else 0
    pin_memory = ARGS["device"].type == "cuda"
    # 遍历三个数据集，创建对应加载器
    for key in ["train", "valid", "test"]:
        drug, mut, cell, labels = standardized_data[key]
        # 初始化数据集
        dataset = DrugSensitivityDataset(drug, mut, cell, labels)
        # 初始化数据加载器
        loader = DataLoader(
            dataset,
            batch_size=ARGS["batch_size"], #批次大小
            shuffle=True if key == "train" else False, #训练集打乱数据，防止过拟合
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False #单独处理最后一个不完整的批次
        )
        loaders[key] = loader
    return loaders


#6. 训练与验证函数
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0 #总损失
    total_pred_loss = 0.0 #总分类损失
    correct = 0 #预测正确的样本数
    total = 0 #总样本数
    #遍历训练批次
    for batch in dataloader:
        drug_feat, mut_feat, cell_feat, labels = [x.to(device) for x in batch]
        #前向传播，得到预测结果和重构特征
        pred_prob, drug_recon, mut_recon = model(drug_feat, mut_feat, cell_feat)
        #计算损失
        total_loss_batch, pred_loss_batch, _, _ = loss_fn(
            pred_prob, drug_recon, mut_recon, drug_feat, mut_feat, labels
        )

        optimizer.zero_grad() #清空梯度
        total_loss_batch.backward() #反向传播，计算梯度
        optimizer.step() #更新模型参数
        #统计损失和准确率
        batch_size = drug_feat.size(0)
        total_loss += total_loss_batch.item() * batch_size
        total_pred_loss += pred_loss_batch.item() * batch_size
        #根据阈值生成二分类标签
        pred_label = (pred_prob.squeeze() >= ARGS["threshold"]).long()
        #统计正确预测数
        correct += (pred_label == labels).sum().item()
        total += batch_size
    #计算平均损失和准确率
    avg_total_loss = total_loss / total
    avg_pred_loss = total_pred_loss / total
    acc = correct / total
    return avg_total_loss, avg_pred_loss, acc

#单轮验证测试函数
def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    correct = 0
    total = 0
    all_pred_probs = [] #储存所有的预测概率
    all_true_labels = [] #储存所有真实标签

    with torch.no_grad():
        for batch in dataloader:
            drug_feat, mut_feat, cell_feat, labels = [x.to(device) for x in batch]

            pred_prob, drug_recon, mut_recon = model(drug_feat, mut_feat, cell_feat)

            total_loss_batch, pred_loss_batch, _, _ = loss_fn(
                pred_prob, drug_recon, mut_recon, drug_feat, mut_feat, labels
            )

            batch_size = drug_feat.size(0)
            total_loss += total_loss_batch.item() * batch_size
            total_pred_loss += pred_loss_batch.item() * batch_size
            pred_label = (pred_prob.squeeze() >= ARGS["threshold"]).long()
            correct += (pred_label == labels).sum().item()
            total += batch_size
            #保存概率和标签，用于后续可视化
            all_pred_probs.extend(pred_prob.squeeze().cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    avg_total_loss = total_loss / total
    avg_pred_loss = total_pred_loss / total
    acc = correct / total
    return avg_total_loss, avg_pred_loss, acc, np.array(all_pred_probs), np.array(all_true_labels)


#7. 结果可视化函数
def plot_train_history(train_history, save_path):
    if len(train_history["total_loss"]) == 0:
        print("Warning: Training history is empty, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6)) #一行二列
    #左图：损失曲线
    axes[0].plot(train_history["total_loss"], label="Training Total Loss", color="blue")
    axes[0].plot(train_history["val_total_loss"], label="Validation Total Loss", color="red")
    axes[0].plot(train_history["pred_loss"], label="Training Classification Loss", color="blue", linestyle="--")
    axes[0].plot(train_history["val_pred_loss"], label="Validation Classification Loss", color="red", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss Value")
    axes[0].set_title("Training and Validation Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图准确率曲线
    axes[1].plot(train_history["acc"], label="Training Accuracy", color="blue")
    axes[1].plot(train_history["val_acc"], label="Validation Accuracy", color="red")
    # Fix: Handle case where val_acc is empty
    if len(train_history["val_acc"]) > 0:
        best_epoch = np.argmax(train_history["val_acc"])
        best_acc = train_history["val_acc"][best_epoch]
        axes[1].scatter(best_epoch, best_acc,
                        color="green", s=100, label=f"Best Val Accuracy: {best_acc:.4f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "train_history.png"), dpi=300, bbox_inches="tight")
    plt.close()

#绘制ROC曲线并计算AUC
def plot_roc_curve(true_labels, pred_probs, save_path):
    if len(np.unique(true_labels)) < 2:
        print("Warning: Only one class present in true labels, skipping ROC curve")
        return 0.0
    #计算ROC参数曲线
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr) #计算AUC

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Drug Sensitivity Prediction ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return roc_auc

# 绘制混淆矩阵热力图
def plot_confusion_matrix(conf_matrix, class_names, save_path):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Drug Sensitivity Prediction Confusion Matrix")
    plt.colorbar(im)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    #添加标签
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], "d"),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

# 绘制预测概率分布直方图
def plot_prob_distribution(true_labels, pred_probs, save_path):
    if len(true_labels) == 0 or len(np.unique(true_labels)) < 2:
        print("Warning: Insufficient data for probability distribution plot")
        return
    # 分离两类样本的预测概率
    sensitive_probs = pred_probs[true_labels == 1]
    non_sensitive_probs = pred_probs[true_labels == 0]

    plt.figure(figsize=(12, 6))
    plt.hist(non_sensitive_probs, bins=30, alpha=0.5, label="True Non-Sensitive (0)", color="red", density=True)
    plt.hist(sensitive_probs, bins=30, alpha=0.5, label="True Sensitive (1)", color="green", density=True)
    # 绘制分类阈值线
    plt.axvline(x=ARGS["threshold"], color="black", linestyle="--",
                label=f"Classification Threshold ({ARGS['threshold']})")
    plt.xlabel("Prediction Probability (0~1)")
    plt.ylabel("Probability Density")
    plt.title("Drug Sensitivity Prediction Probability Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "prob_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


#8. 主训练流程
def main():
    #步骤1：加载并预处理数据
    print("=" * 50)
    print("1. Loading and Preprocessing Data")
    print("=" * 50)
    raw_data = load_data(FILES)
    processed_data, ic50_threshold = ic50_to_binary(raw_data)
    standardized_data, scaler_dict = get_standardized_features(processed_data)
    #步骤2：创建数据加载器
    print("\n2. Creating DataLoaders")
    dataloaders = create_dataloaders(standardized_data)
    print(f"Training batches: {len(dataloaders['train'])}")
    print(f"Validation batches: {len(dataloaders['valid'])}")
    print(f"Test batches: {len(dataloaders['test'])}")
    #步骤3：初始化模型和训练组件
    print("\n" + "=" * 50)
    print("3. Initializing Model and Training Components")
    print("=" * 50)
    train_drug_dim = standardized_data["train"][0].shape[1]
    train_mut_dim = standardized_data["train"][1].shape[1]
    train_cell_dim = standardized_data["train"][2].shape[1]
    # 分类网络输入维度 = 药物编码(64) + 突变编码(64) + 细胞特征
    forward_input_dim = ARGS["code_dim"] * 2 + train_cell_dim
    print(
        f"Drug feature dim: {train_drug_dim}, Mutation feature dim: {train_mut_dim}, Cell feature dim: {train_cell_dim}")
    print(f"Forward network input dim: {forward_input_dim}")
    # 初始化药物自编码器
    drug_autoencoder = DeepAutoencoderThreeHiddenLayers(
        input_dim=train_drug_dim,
        hidden_dims=ARGS["drug_hidden_dims"],
        code_dim=ARGS["code_dim"],
        dropout=True,
        dropout_rate=ARGS["dropout_rate"]
    )
    # 初始化突变自编码器
    mut_autoencoder = DeepAutoencoderThreeHiddenLayers(
        input_dim=train_mut_dim,
        hidden_dims=ARGS["mut_hidden_dims"],
        code_dim=ARGS["code_dim"],
        dropout=True,
        dropout_rate=ARGS["dropout_rate"]
    )
    # 初始化分类网络
    forward_network = ForwardNetworkTwoHiddenLayers(
        input_dim=forward_input_dim,
        hidden_dim1=ARGS["forward_hidden1"],
        hidden_dim2=ARGS["forward_hidden2"]
    )
    #初始化主模型
    model = DEERS_Concat(drug_autoencoder, mut_autoencoder, forward_network).to(ARGS["device"])
    loss_fn = MergedBCELoss()
    # 优化器使用Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS["lr"], weight_decay=1e-5)
    #验证损失不下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)
    # 初始化早停机制
    early_stopping = EarlyStopping(
        patience=ARGS["early_stop_patience"],
        verbose=True,
        save_path=os.path.join(RESULT_DIR, "best_model.pth")
    )
    #步骤4：开始训练
    print("\n" + "=" * 50)
    print(f"4. Starting Training (Device: {ARGS['device']}, Total Epochs: {ARGS['epochs']})")
    print("=" * 50)
    train_history = {
        "total_loss": [], "pred_loss": [], "acc": [],
        "val_total_loss": [], "val_pred_loss": [], "val_acc": []
    }

    for epoch in range(1, ARGS["epochs"] + 1):
        print(f"\nEpoch {epoch}/{ARGS['epochs']}")
        print("-" * 30)
        #训练一轮
        train_total_loss, train_pred_loss, train_acc = train_epoch(
            model, dataloaders["train"], loss_fn, optimizer, ARGS["device"]
        )
        #验证一轮
        val_total_loss, val_pred_loss, val_acc, _, _ = validate_epoch(
            model, dataloaders["valid"], loss_fn, ARGS["device"]
        )

        scheduler.step(val_total_loss) #更新学习率
        #保存训练历史
        train_history["total_loss"].append(train_total_loss)
        train_history["pred_loss"].append(train_pred_loss)
        train_history["acc"].append(train_acc)
        train_history["val_total_loss"].append(val_total_loss)
        train_history["val_pred_loss"].append(val_pred_loss)
        train_history["val_acc"].append(val_acc)

        print(
            f"Training: Total Loss={train_total_loss:.4f}, Classification Loss={train_pred_loss:.4f}, Accuracy={train_acc:.4f}")
        print(
            f"Validation: Total Loss={val_total_loss:.4f}, Classification Loss={val_pred_loss:.4f}, Accuracy={val_acc:.4f}")
        #曹婷判断
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered! Stopping training at Epoch {epoch}")
            break
    #保存训练历史
    np.save(os.path.join(RESULT_DIR, "train_history.npy"), train_history)
    print(f"\nTraining completed! Training history saved to: {os.path.join(RESULT_DIR, 'train_history.npy')}")
    #步骤5：步骤5：测试集评估
    print("\n" + "=" * 50)
    print("5. Evaluating on Test Set (Loading Best Model)")
    print("=" * 50)
    model = early_stopping.load_best_model(model) #加载最优权重
    model.to(ARGS["device"])
    #测试集推理
    test_total_loss, test_pred_loss, test_acc, test_pred_probs, test_true_labels = validate_epoch(
        model, dataloaders["test"], loss_fn, ARGS["device"]
    )
    #生成测试集预测标签和评估指标
    test_pred_labels = (test_pred_probs >= ARGS["threshold"]).astype(int)
    conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
    class_report = classification_report(
        test_true_labels, test_pred_labels, target_names=["Non-Sensitive (0)", "Sensitive (1)"], output_dict=True
    )
    #保存测试结果
    test_results = {
        "test_acc": test_acc,
        "test_total_loss": test_total_loss,
        "test_pred_loss": test_pred_loss,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "pred_probs": test_pred_probs,
        "true_labels": test_true_labels,
        "threshold": ARGS["threshold"],
        "ic50_binary_threshold": ic50_threshold
    }
    np.save(os.path.join(RESULT_DIR, "test_results.npy"), test_results)

    print(f"\nTest Set Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Classification Loss: {test_pred_loss:.4f}")
    print(f"Confusion Matrix:")
    print(conf_matrix)
    print(f"Classification Report:")
    print(
        classification_report(test_true_labels, test_pred_labels, target_names=["Non-Sensitive (0)", "Sensitive (1)"]))
    #步骤6：生成可视化图表
    print("\n" + "=" * 50)
    print("6. Generating Visualization Plots")
    print("=" * 50)
    plot_train_history(train_history, RESULT_DIR)
    roc_auc = plot_roc_curve(test_true_labels, test_pred_probs, RESULT_DIR)
    plot_confusion_matrix(conf_matrix, ["Non-Sensitive (0)", "Sensitive (1)"], RESULT_DIR)
    plot_prob_distribution(test_true_labels, test_pred_probs, RESULT_DIR)
    print(f"\nVisualization plots saved to: {RESULT_DIR}")
    print(f"ROC Curve AUC Value: {roc_auc:.4f}")
    #步骤7：输出文件清单
    print("\n" + "=" * 50)
    print("7. Output File List")
    print("=" * 50)
    output_files = [
        "best_model.pth (Best Model Weights)",
        "train_history.npy (Training History)",
        "test_results.npy (Test Set Results)",
        "scaler_dict.npy (Feature Scalers)",
        "train_history.png (Training Curves)",
        "roc_curve.png (ROC Curve)",
        "confusion_matrix.png (Confusion Matrix)",
        "prob_distribution.png (Probability Distribution)"
    ]
    for idx, file in enumerate(output_files, 1):
        print(f"{idx}. {file}")

    print(f"\nAll results saved to: {RESULT_DIR}")

#主函数入口
if __name__ == "__main__":
    main()