import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ====================== 0. 固定所有随机种子 ==================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ====================== 1. 固定配置 =========================================
ARGS = {
    "code_dim": 64,
    "drug_hidden_dims": [256, 128],
    "mut_hidden_dims": [256, 128],
    "forward_hidden1": 128,
    "forward_hidden2": 64,
    "threshold": 0.5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

MODEL_PATH = r"E:\大创\code&web\model_code\result\best_model.pth"          
SCALER_PATH = r"E:\大创\code&web\model_code\result\scaler_dict.npy"        

# ====================== 2. 全局变量（模型和标准化器缓存） ======================
model = None
drug_scaler = None
mut_scaler = None
cell_scaler = None

# ====================== 3. 模型结构定义 =======================================
class DeepAutoencoderThreeHiddenLayers(nn.Module):
    def __init__(self, input_dim, hidden_dims, code_dim):
        super().__init__()
        # 编码器：输入→隐藏层→编码层
        encoder_modules = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(0.3)]
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            encoder_modules.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.3)])
        encoder_modules.append(nn.Linear(hidden_dims[-1], code_dim))
        self.encoder = nn.Sequential(*encoder_modules)

        # 解码器：编码层→隐藏层→输出（与输入维度一致）
        decoder_modules = [nn.Linear(code_dim, hidden_dims[-1]), nn.ReLU(), nn.Dropout(0.3)]
        for in_dim, out_dim in zip(hidden_dims[::-1][:-1], hidden_dims[::-1][1:]):
            decoder_modules.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.3)])
        decoder_modules.extend([nn.Linear(hidden_dims[0], input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return code, recon

class ForwardNetworkTwoHiddenLayers(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predictor(x)

class DEERS_Concat(nn.Module):
    def __init__(self, drug_autoenc, mut_autoenc, forward_net):
        super().__init__()
        self.drug_autoenc = drug_autoenc  # 药物自编码器
        self.mut_autoenc = mut_autoenc    # 突变自编码器
        self.forward_net = forward_net    # 预测网络

    def forward(self, drug_feat, mut_feat, cell_feat):
        # 提取药物、突变的编码特征
        drug_code, _ = self.drug_autoenc(drug_feat)
        mut_code, _ = self.mut_autoenc(mut_feat)
        # 拼接特征（药物编码+突变编码+细胞富集特征）
        concat_feat = torch.cat([drug_code, mut_code, cell_feat], dim=1)
        # 预测敏感性概率
        pred_prob = self.forward_net(concat_feat)
        return pred_prob

# ====================== 4. 核心函数：3-CSV预测 =================================
def predict_three_csv(mut_csv, smile_csv, ssgsea_csv, output_csv, task_id=None, update_progress_func=None):
    """
    从3个输入CSV预测药物敏感性，输出CSV结果（结果100%可复现）
    :param mut_csv: 细胞突变数据CSV路径（需含cell_idx、cell_line列）
    :param smile_csv: 药物SMILE特征CSV路径（需含drug_idx、drug_name列）
    :param ssgsea_csv: 细胞ssGSEA富集CSV路径（需含cell_idx、cell_name列）
    :param output_csv: 预测结果输出CSV路径
    :param task_id: 任务ID（用于关联前端进度跟踪，可选）
    :param update_progress_func: 进度更新函数（格式：func(task_id, step, percentage, msg)，可选）
    :return: 预测成功返回True，失败返回False
    """
    global model, drug_scaler, mut_scaler, cell_scaler
    set_seed(42)
    
    try:
        # -------------------------- 步骤1：读取并验证输入CSV（进度：60%-65%）
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 60, "开始读取3个CSV文件...")
        
        # 读取数据时指定dtype，避免隐式类型转换导致的差异
        mut_df = pd.read_csv(mut_csv, dtype={"cell_idx": int, "cell_line": str})
        smile_df = pd.read_csv(smile_csv, dtype={"drug_idx": int, "drug_name": str})
        ssgsea_df = pd.read_csv(ssgsea_csv, dtype={"cell_idx": int, "cell_name": str})
        
        # 验证必填列
        required_cols = {
            "突变CSV": ["cell_idx", "cell_line"],
            "药物CSV": ["drug_idx", "drug_name"],
            "富集CSV": ["cell_idx", "cell_name"]
        }
        if not all(col in mut_df.columns for col in required_cols["突变CSV"]):
            err_msg = "突变CSV缺少必填列：cell_idx 或 cell_line"
            if task_id and update_progress_func:
                update_progress_func(task_id, 4, 65, f"错误：{err_msg}")
            print(f"预测失败：{err_msg}")
            return False
        
        if not all(col in smile_df.columns for col in required_cols["药物CSV"]):
            err_msg = "药物CSV缺少必填列：drug_idx 或 drug_name"
            if task_id and update_progress_func:
                update_progress_func(task_id, 4, 65, f"错误：{err_msg}")
            print(f"预测失败：{err_msg}")
            return False
        
        if not all(col in ssgsea_df.columns for col in required_cols["富集CSV"]):
            err_msg = "富集CSV缺少必填列：cell_idx 或 cell_name"
            if task_id and update_progress_func:
                update_progress_func(task_id, 4, 65, f"错误：{err_msg}")
            print(f"预测失败：{err_msg}")
            return False
        
        # 验证特征列非空
        if mut_df.shape[1] <= 2 or smile_df.shape[1] <= 2 or ssgsea_df.shape[1] <= 2:
            err_msg = "某个CSV文件缺少特征列（需在索引列后添加至少1列特征数据）"
            if task_id and update_progress_func:
                update_progress_func(task_id, 4, 65, f"错误：{err_msg}")
            print(f"预测失败：{err_msg}")
            return False
        
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 65, f"CSV文件读取完成（突变{len(mut_df)}行/药物{len(smile_df)}行/富集{len(ssgsea_df)}行）")

        # -------------------------- 步骤2：加载标准化器 ============================
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 68, "正在加载标准化器...")
        
        # 加载标准化器
        if drug_scaler is None or mut_scaler is None or cell_scaler is None:
            try:
                scaler_dict = np.load(SCALER_PATH, allow_pickle=True).item()
                drug_scaler = scaler_dict["drug"]
                mut_scaler = scaler_dict["mut"]
                cell_scaler = scaler_dict["cell"]
                drug_scaler.random_state = 42
                mut_scaler.random_state = 42
                cell_scaler.random_state = 42
            except Exception as e:
                err_msg = f"标准化器加载失败：{str(e)}（检查SCALER_PATH路径是否正确）"
                if task_id and update_progress_func:
                    update_progress_func(task_id, 4, 70, f"错误：{err_msg}")
                print(f"预测失败：{err_msg}")
                return False
        
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 70, "正在初始化预测模型...")

        # -------------------------- 步骤3：加载模型（强制评估模式） ======================
        if model is None:
            try:
                # 计算特征维度
                drug_feat_dim = smile_df.shape[1] - 2
                mut_feat_dim = mut_df.shape[1] - 2
                cell_feat_dim = ssgsea_df.shape[1] - 2
                forward_input_dim = ARGS["code_dim"] * 2 + cell_feat_dim

                # 初始化模型
                drug_autoenc = DeepAutoencoderThreeHiddenLayers(
                    input_dim=drug_feat_dim,
                    hidden_dims=ARGS["drug_hidden_dims"],
                    code_dim=ARGS["code_dim"]
                )
                mut_autoenc = DeepAutoencoderThreeHiddenLayers(
                    input_dim=mut_feat_dim,
                    hidden_dims=ARGS["mut_hidden_dims"],
                    code_dim=ARGS["code_dim"]
                )
                forward_net = ForwardNetworkTwoHiddenLayers(
                    input_dim=forward_input_dim,
                    hidden1=ARGS["forward_hidden1"],
                    hidden2=ARGS["forward_hidden2"]
                )

                # 组合模型并加载权重
                model = DEERS_Concat(drug_autoenc, mut_autoenc, forward_net)
                # 加载权重时严格匹配，避免意外
                state_dict = torch.load(MODEL_PATH, map_location=ARGS["device"], weights_only=True)
                model.load_state_dict(state_dict, strict=True)
                
                model.eval()
                model.to(ARGS["device"])
                for param in model.parameters():
                    param.requires_grad = False

            except Exception as e:
                err_msg = f"模型加载失败请点击重试：{str(e)}（检查MODEL_PATH路径/模型结构是否与训练时一致）"
                if task_id and update_progress_func:
                    update_progress_func(task_id, 4, 75, f"错误：{err_msg}")
                print(f"预测失败请点击重试：{err_msg}")
                return False
        
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 75, "模型和标准化器加载完成")

        # -------------------------- 步骤4：生成药物-细胞配对 =======================
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 76, "正在生成药物-细胞预测配对...")
        
        # 筛选有效索引并排序
        valid_cell_idx = sorted(list(set(mut_df["cell_idx"]) & set(ssgsea_df["cell_idx"])))
        valid_drug_idx = sorted(list(set(smile_df["drug_idx"])))
        
        if len(valid_cell_idx) == 0:
            err_msg = "突变CSV和富集CSV无共同的cell_idx（无有效细胞可预测）"
            if task_id and update_progress_func:
                update_progress_func(task_id, 4, 78, f"错误：{err_msg}")
            print(f"预测失败：{err_msg}")
            return False
        
        if len(valid_drug_idx) == 0:
            err_msg = "药物CSV无有效drug_idx（无有效药物可预测）"
            if task_id and update_progress_func:
                update_progress_func(task_id, 4, 78, f"错误：{err_msg}")
            print(f"预测失败：{err_msg}")
            return False
        
        # 生成配对
        pair_list = []
        for drug_idx in valid_drug_idx:
            for cell_idx in valid_cell_idx:
                pair_list.append({"drug_idx": drug_idx, "cell_idx": cell_idx})
        pair_df = pd.DataFrame(pair_list)
        total_pairs = len(pair_df)
        
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 80, f"生成{total_pairs}条药物-细胞配对（药物{len(valid_drug_idx)}种/细胞{len(valid_cell_idx)}种）")

        # -------------------------- 步骤5：批量预测 ===========================
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 80, "开始批量预测（结果100%可复现）...")
        
        # 构建特征映射
        drug_feat_map = {k: v for k, v in sorted(zip(smile_df["drug_idx"], smile_df.iloc[:, 2:].values))}
        mut_feat_map = {k: v for k, v in sorted(zip(mut_df["cell_idx"], mut_df.iloc[:, 2:].values))}
        cell_feat_map = {k: v for k, v in sorted(zip(ssgsea_df["cell_idx"], ssgsea_df.iloc[:, 2:].values))}
        drug_name_map = {k: v for k, v in sorted(zip(smile_df["drug_idx"], smile_df["drug_name"]))}
        cell_line_map = {k: v for k, v in sorted(zip(mut_df["cell_idx"], mut_df["cell_line"]))}

        # 批量预测配置
        batch_size = 32
        total_batches = (total_pairs + batch_size - 1) // batch_size
        predict_results = []

        # 禁用梯度计算
        with torch.no_grad():
            for batch_idx in range(total_batches):
                # 计算进度
                batch_progress = 80 + (batch_idx + 1) / total_batches * 18
                current_count = min((batch_idx + 1) * batch_size, total_pairs)
                if task_id and update_progress_func:
                    update_progress_func(
                        task_id, 4, round(batch_progress, 1),
                        f"预测批次 {batch_idx+1}/{total_batches}（{current_count}/{total_pairs}条）"
                    )

                # 提取批次数据
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_pairs)
                batch_pairs = pair_df.iloc[batch_start:batch_end]
                batch_drug_idx = batch_pairs["drug_idx"].tolist()
                batch_cell_idx = batch_pairs["cell_idx"].tolist()

                # 准备特征
                batch_drug_feat = []
                batch_mut_feat = []
                batch_cell_feat = []
                for d_idx, c_idx in zip(batch_drug_idx, batch_cell_idx):
                    # 药物特征
                    drug_feat = drug_feat_map[d_idx].reshape(1, -1).astype(np.float32)
                    drug_feat_norm = drug_scaler.transform(drug_feat)
                    batch_drug_feat.append(drug_feat_norm)
                    
                    # 突变特征
                    mut_feat = mut_feat_map[c_idx].reshape(1, -1).astype(np.float32)
                    mut_feat_norm = mut_scaler.transform(mut_feat)
                    batch_mut_feat.append(mut_feat_norm)
                    
                    # 细胞特征
                    cell_feat = cell_feat_map[c_idx].reshape(1, -1).astype(np.float32)
                    cell_feat_norm = cell_scaler.transform(cell_feat)
                    batch_cell_feat.append(cell_feat_norm)

                # 转换为Tensor
                drug_tensor = torch.tensor(np.concatenate(batch_drug_feat), dtype=torch.float32, device=ARGS["device"])
                mut_tensor = torch.tensor(np.concatenate(batch_mut_feat), dtype=torch.float32, device=ARGS["device"])
                cell_tensor = torch.tensor(np.concatenate(batch_cell_feat), dtype=torch.float32, device=ARGS["device"])

                # 模型预测
                pred_prob_tensor = model(drug_tensor, mut_tensor, cell_tensor)
                pred_prob = pred_prob_tensor.squeeze().cpu().numpy()
                if len(batch_pairs) == 1:
                    pred_prob = np.array([pred_prob])

                # 整理结果
                for idx in range(len(batch_pairs)):
                    d_idx = batch_drug_idx[idx]
                    c_idx = batch_cell_idx[idx]
                    prob = round(float(pred_prob[idx]), 6)
                    label = 1 if prob >= ARGS["threshold"] else 0
                    predict_results.append({
                        "drug_idx": d_idx,
                        "drug_name": drug_name_map[d_idx],
                        "cell_idx": c_idx,
                        "cell_line": cell_line_map[c_idx],
                        "预测敏感性": "高" if label == 1 else "低",
                        "预测概率": prob,
                        "预测标签": label
                    })

        # -------------------------- 步骤6：保存结果 ============================
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 98, "正在保存预测结果到CSV...")
        
        # 保存结果
        result_df = pd.DataFrame(predict_results)
        # 强制排序结果
        result_df = result_df.sort_values(by=["drug_idx", "cell_idx"]).reset_index(drop=True)
        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
        
        # 验证结果文件
        if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
            err_msg = "结果CSV生成失败（可能是磁盘空间不足或权限不够）"
            if task_id and update_progress_func:
                update_progress_func(task_id, 4, 99, f"错误：{err_msg}")
            print(f"预测失败：{err_msg}")
            return False
        
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 100, f"预测完成！共生成{len(result_df)}条结果（结果可复现）")
        
        print(f"预测成功！结果已保存至：{output_csv}（{len(result_df)}条记录）")
        return True

    except Exception as e:
        err_msg = f"预测过程异常请点击重试：{str(e)}"
        if task_id and update_progress_func:
            update_progress_func(task_id, 4, 90, f"错误：{err_msg}")
        print(f"预测失败请点击重试：{err_msg}")
        return False