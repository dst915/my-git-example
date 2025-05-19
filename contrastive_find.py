import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.feature_bagging import FeatureBagging
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from tabulate import tabulate
import joblib
import matplotlib.pyplot as plt
import datetime
import uuid
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ==== 用户配置区域 ====
use_contrastive = True
contaminations = [0.005, 0.01]
contamination_min = 0.0025
contamination_max = 0.02
algorithms = [
    ("Isolation Forest", IForest),
    ("LOF", LOF),
    ("KNN", KNN),
    ("HBOS", HBOS),
    ("Feature Bagging", lambda **kw: FeatureBagging(base_estimator=LOF(), **kw))
]
error_csv = "error.csv"
# ====================

# --------- 工具类和基础函数 ---------
class HtmlLogger:
    """HTML日志记录器，支持表格、图片、代码等写入"""
    def __init__(self, filepath, title=''):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.file.write(f'<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>{title}</title>')
        self.file.write('<style>body{font-family:Arial,Helvetica,sans-serif;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:8px;} th{background:#f2f2f2;} pre{background:#f8f8f8;padding:8px;}</style></head><body>')
        if title:
            self.file.write(f'<h1>{title}</h1>\n')

    def write(self, html):
        self.file.write(html)
        self.file.flush()

    def write_table(self, df, max_rows=20, title=None):
        if title:
            self.file.write(f'<h2>{title}</h2>')
        if len(df) > max_rows:
            self.file.write(f"<p>显示前{max_rows}行，共{len(df)}行</p>")
            df = df.head(max_rows)
        self.file.write(df.to_html(index=True, escape=False))
        self.file.flush()

    def write_tabulate(self, df, title=None):
        if title:
            self.file.write(f'<h2>{title}</h2>')
        html_table = tabulate(df, headers='keys', tablefmt='html', showindex=False)
        self.file.write(html_table)
        self.file.flush()

    def write_img(self, img_path, caption=None, width=600):
        if caption:
            self.file.write(f'<h3>{caption}</h3>')
        self.file.write(f'<img src="{os.path.basename(img_path)}" width="{width}"/><br>')

    def write_code(self, code, lang='python'):
        self.file.write(f'<pre><code class="{lang}">{code}</code></pre>')

    def close(self):
        self.file.write('</body></html>')
        self.file.close()

def read_table(file):
    """读取表格文件，支持csv/xlsx/xls/npy/parquet"""
    if file is None or not os.path.exists(file):
        return None
    ext = os.path.splitext(file)[-1].lower()
    if ext == '.csv':
        return pd.read_csv(file)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file)
    elif ext == '.npy':
        arr = np.load(file)
        return pd.DataFrame(arr)
    elif ext == '.parquet':
        return pd.read_parquet(file)
    else:
        raise ValueError(f"不支持的文件格式: {file}")

def preprocess_df(df):
    """对DataFrame做独热编码和布尔类型转换"""
    obj_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(obj_cols) > 0:
        df = pd.get_dummies(df, columns=obj_cols, dummy_na=True)
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    return df

def add_stat_features(df):
    """为DataFrame添加统计特征（行均值、方差、缺失数等）"""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df['row_mean'] = df[num_cols].mean(axis=1)
    df['row_std'] = df[num_cols].std(axis=1)
    df['row_max'] = df[num_cols].max(axis=1)
    df['row_min'] = df[num_cols].min(axis=1)
    df['nan_count'] = df.isna().sum(axis=1)
    df['nan_ratio'] = df.isna().sum(axis=1) / df.shape[1]
    df['has_nan'] = (df.isna().sum(axis=1) > 0).astype(int)
    col_uniques = []
    col_modes = []
    col_mode_ratios = []
    for col in df.columns:
        uniques = df[col].nunique()
        col_uniques.append(uniques)
        m, count = mode(df[col].dropna(), nan_policy='omit')
        if hasattr(m, '__len__') and len(m) > 0:
            modeval = m[0]
            modecount = count[0]
        else:
            modeval = m
            modecount = count
        if pd.isna(modeval):
            col_modes.append(np.nan)
            col_mode_ratios.append(np.nan)
        else:
            mode_ratio = modecount / max(1, df[col].notna().sum())
            col_modes.append(modeval)
            col_mode_ratios.append(mode_ratio)
    for idx, col in enumerate(df.columns):
        df[f'{col}_nunique'] = col_uniques[idx]
        df[f'{col}_mode_ratio'] = col_mode_ratios[idx]
    for col in num_cols:
        col_data = df[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        iqr_mask = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).astype(int)
        df[f'{col}_iqr_outlier'] = iqr_mask
        mean = col_data.mean()
        std = col_data.std()
        if std > 0:
            zscore_mask = ((np.abs(col_data - mean) > 3 * std)).astype(int)
        else:
            zscore_mask = 0
        df[f'{col}_zscore_outlier'] = zscore_mask
    return df

def prepare_data(args):
    """读取、预处理数据并组合成训练集和测试集"""
    df_normal = read_table(args.normal)
    df_abnormal = read_table(args.abnormal)
    df_test = read_table(args.test)
    if df_normal is not None:
        df_normal = preprocess_df(df_normal)
        df_normal = add_stat_features(df_normal)
        df_normal = df_normal.fillna(0)
    if df_abnormal is not None:
        df_abnormal = preprocess_df(df_abnormal)
        df_abnormal = add_stat_features(df_abnormal)
        df_abnormal = df_abnormal.fillna(0)
    if df_test is not None:
        df_test = preprocess_df(df_test)
        df_test = add_stat_features(df_test)
        df_test = df_test.fillna(0)
    # 只输入异常样本训练
    if getattr(args, "abnormal_only", False) and df_abnormal is not None:
        X = df_abnormal.values
        y_true = np.ones(len(df_abnormal), dtype=int)
        df_merged = df_abnormal
        return X, y_true, df_merged, None, df_abnormal, None
    if df_normal is None and df_abnormal is None and df_test is not None:
        X = df_test.values
        y_true = np.zeros(len(df_test), dtype=int)
        df_merged = df_test
    elif df_normal is not None and df_abnormal is not None:
        X = np.vstack([df_normal.values, df_abnormal.values])
        y_true = np.array([0]*len(df_normal) + [1]*len(df_abnormal))
        df_merged = pd.concat([df_normal, df_abnormal], ignore_index=True)
    elif df_normal is not None and df_abnormal is None:
        X = df_normal.values
        y_true = np.zeros(len(df_normal), dtype=int)
        df_merged = df_normal
    elif df_normal is not None and df_test is not None:
        X = np.vstack([df_normal.values, df_test.values])
        y_true = np.array([0]*len(df_normal) + [0]*len(df_test))
        df_merged = pd.concat([df_normal, df_test], ignore_index=True)
    else:
        raise RuntimeError("输入文件组合不支持！")
    return X, y_true, df_merged, df_normal, df_abnormal, df_test

def get_scaler(X, scaler_path):
    """获取或训练标准化器"""
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, scaler_path)
    return scaler

def compute_specificity_fpr(y_true, y_pred):
    """计算特异度和FPR"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    return specificity, fpr

def compute_all_metrics(y_true, y_pred, y_score=None):
    """输出各类二分类指标"""
    acc = accuracy_score(y_true, y_pred)
    specificity, fpr = compute_specificity_fpr(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_true)) > 1 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "准确率": f"{acc:.4f}",
        "查准率(Precision)": f"{precision:.4f}",
        "查全率(Recall)": f"{recall:.4f}",
        "F1": f"{f1:.4f}",
        "AUC": f"{auc:.4f}",
        "特异度(Specificity)": f"{specificity:.4f}",
        "FPR": f"{fpr:.4f}",
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "检测异常数": int((y_pred == 1).sum())
    }

# --------- SimCLR对比学习编码器 ---------
class SimCLR_MLPEncoder(nn.Module):
    """SimCLR风格的MLP特征编码器"""
    def __init__(self, input_dim, feature_dim=32, projection_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(64, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, feature_dim)
        )
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

def simclr_augment(x):
    """SimCLR风格数据增强"""
    x1 = x + torch.randn_like(x) * 0.05
    x2 = x + torch.randn_like(x) * 0.05
    mask1 = (torch.rand_like(x1) > 0.1).float()
    mask2 = (torch.rand_like(x2) > 0.1).float()
    return x1 * mask1, x2 * mask2

def simclr_loss(z1, z2, temperature=0.5):
    """SimCLR对比损失"""
    z1 = nn.functional.normalize(z1, p=2, dim=1)
    z2 = nn.functional.normalize(z2, p=2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    positives = torch.cat([sim_i_j, sim_j_i], dim=0)
    mask = (~torch.eye(2*N, dtype=bool)).to(z.device)
    negatives = sim[mask].view(2*N, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2*N, dtype=torch.long).to(z.device)
    return nn.CrossEntropyLoss()(logits, labels)

def train_encoder(X_scaled, encoder_path, feature_dim=32, epochs=50, batch_size=128, out_dir=None, logger=None):
    """训练SimCLR编码器"""
    loss_history = []
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    encoder = SimCLR_MLPEncoder(X_scaled.shape[1], feature_dim=feature_dim)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_tensor.size(0))
        losses = []
        for i in range(0, X_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            x = X_tensor[idx]
            x1, x2 = simclr_augment(x)
            z1 = encoder(x1)
            z2 = encoder(x2)
            loss = simclr_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = np.mean(losses)
        loss_history.append(epoch_loss)
        if (epoch+1) % 5 == 0 or epoch == 0:
            if logger:
                logger.write(f"<p>Epoch {epoch+1}, Loss: {epoch_loss:.4f}</p>\n")
            else:
                print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    torch.save(encoder.state_dict(), encoder_path)
    encoder.eval()
    if out_dir:
        plt.figure(figsize=(8,4))
        plt.plot(np.arange(1, epochs+1), loss_history, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('SimCLR Loss')
        plt.title('SimCLR Contrastive Loss Curve')
        plt.grid(True)
        plt.tight_layout()
        loss_curve_path = os.path.join(out_dir, "simclr_loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        if logger:
            logger.write_img(loss_curve_path, caption="SimCLR对比学习损失曲线")
    return encoder

def get_encoder(X_scaled, encoder_path, feature_dim=32, out_dir=None, logger=None):
    """获取或训练SimCLR编码器"""
    if os.path.exists(encoder_path):
        encoder = SimCLR_MLPEncoder(X_scaled.shape[1], feature_dim=feature_dim)
        encoder.load_state_dict(torch.load(encoder_path))
        encoder.eval()
    else:
        encoder = train_encoder(X_scaled, encoder_path, feature_dim=feature_dim, out_dir=out_dir, logger=logger)
    return encoder

# --------- 模型训练与融合投票 ---------
def fit_predict_models(X_feat, y_true, algorithms, contamination, model_dir, feat_dim, retrain_models=False):
    """训练或加载每个算法模型并预测"""
    results = {}
    for name, Algo in algorithms:
        model_path = os.path.join(model_dir, f"{name.replace(' ', '_')}_cont{contamination}_feat{feat_dim}.pkl")
        retrain = retrain_models
        if not retrain:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                if hasattr(model, 'n_features_in_'):
                    if model.n_features_in_ != feat_dim:
                        retrain = True
            else:
                retrain = True
        if retrain:
            model = Algo(contamination=contamination)
            model.fit(X_feat)
            joblib.dump(model, model_path)
        y_pred = model.predict(X_feat)
        y_score = model.decision_function(X_feat)
        results[name] = (y_pred, y_score)
    return results

def dynamic_contamination(scores, min_c, max_c):
    """动态计算异常比例"""
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    threshold = mean + 1.5 * std
    idx = np.where(scores >= threshold)[0]
    prop = len(idx) / n
    prop = max(min_c, min(max_c, prop))
    k = int(n * prop)
    if k < 1:
        k = 1
    top_idx = np.argsort(scores)[-k:]
    return top_idx, prop

def fusion_vote(results, contamination, y_true, require_all=False, dynamic_c=False):
    """模型融合投票及指标评估"""
    preds = np.array([results[name][0] for name in results])
    y_score_list = np.array([results[name][1] for name in results])
    y_score_list_norm = []
    for i in range(y_score_list.shape[0]):
        s = y_score_list[i]
        if np.std(s) > 1e-6:
            s_norm = (s - np.mean(s)) / np.std(s)
        else:
            s_norm = s - np.mean(s)
        y_score_list_norm.append(s_norm)
    y_score_list_norm = np.array(y_score_list_norm)
    avg_score = y_score_list_norm.mean(axis=0)
    if dynamic_c:
        top_idx, real_cont = dynamic_contamination(avg_score, contamination_min, contamination_max)
    else:
        max_anomaly = int(len(avg_score) * contamination)
        top_idx = np.argsort(avg_score)[-max_anomaly:]
        real_cont = contamination
    final_y_vote = np.zeros_like(avg_score, dtype=int)
    final_y_vote[top_idx] = 1
    if require_all:
        high_confidence_idx = np.where(preds.sum(axis=0) == preds.shape[0])[0]
    else:
        high_confidence_idx = np.where(preds.sum(axis=0) >= (preds.shape[0] // 2 + 1))[0]
    metrics = []
    for idx, (name, (y_pred, y_score)) in enumerate(results.items()):
        metric = compute_all_metrics(y_true, y_pred, y_score)
        metric["模型"] = name
        metrics.append(metric)
    metric = compute_all_metrics(y_true, final_y_vote, avg_score)
    metric["模型"] = f"融合投票(动态比例{real_cont:.4f})"
    metrics.append(metric)
    return metrics, final_y_vote, high_confidence_idx, avg_score

def retrain_with_pseudo_anomaly(X, df_merged, high_confidence_idx, logger=None):
    """用高置信伪异常反哺训练集"""
    if len(high_confidence_idx) == 0:
        if logger:
            logger.write("<p>无高置信伪异常，跳过伪标签反哺训练。</p>")
        return X, None
    pseudo_anomaly = df_merged.iloc[high_confidence_idx]
    X_new = np.vstack([X, pseudo_anomaly.values])
    y_new = np.array([0]*len(X) + [1]*len(pseudo_anomaly))
    if logger:
        logger.write(f"<p>已将{len(pseudo_anomaly)}条高置信伪异常样本反哺到训练集。</p>")
    return X_new, y_new

def evaluate_models(X_eval, y_eval, algorithms, contamination, model_dir, feat_dim, retrain_models, logger):
    """对伪标签数据进行训练和评估"""
    logger.write("<h2>反哺后半监督训练与评估</h2>")
    results = fit_predict_models(X_eval, y_eval, algorithms, contamination, model_dir, feat_dim, retrain_models=retrain_models)
    preds = np.array([results[name][0] for name in results])
    y_score_list = np.array([results[name][1] for name in results])
    avg_score = y_score_list.mean(axis=0)
    top_idx, real_cont = dynamic_contamination(avg_score, contamination_min, contamination_max)
    final_y_vote = np.zeros_like(avg_score, dtype=int)
    final_y_vote[top_idx] = 1
    metrics = []
    for idx, (name, (y_pred, y_score)) in enumerate(results.items()):
        metric = compute_all_metrics(y_eval, y_pred, y_score)
        metric["模型"] = name
        metrics.append(metric)
    metric = compute_all_metrics(y_eval, final_y_vote, avg_score)
    metric["模型"] = f"融合投票(动态比例{real_cont:.4f})"
    metrics.append(metric)
    df_metrics = pd.DataFrame(metrics)
    logger.write_tabulate(df_metrics, title="各模型性能指标")
    pseudo_idx = np.where(y_eval == 1)[0]
    detected = np.where(final_y_vote == 1)[0]
    detected_pseudo = set(pseudo_idx) & set(detected)
    logger.write(f"<p>伪异常总数: {len(pseudo_idx)}, 被检测出的伪异常数: {len(detected_pseudo)}</p>")
    logger.write(f"<p>被检测出的伪异常样本行号: {sorted(detected_pseudo)}</p>")

def print_anomaly_table_html(df, anomaly_indices, logger, scores=None, title="检测出的异常样本（全部展示）"):
    """将检测出的异常样本写入HTML报告"""
    if len(anomaly_indices) > 0:
        df_anomaly = df.iloc[anomaly_indices].copy()
        if scores is not None:
            df_anomaly['anomaly_score'] = scores[anomaly_indices]
            df_anomaly = df_anomaly.sort_values('anomaly_score', ascending=False)
        logger.write_table(df_anomaly, max_rows=len(df_anomaly), title=title)
        logger.write(f"<p>共检测出 {len(df_anomaly)} 条异常样本，已写入 {error_csv}</p>")
    else:
        logger.write("<p>未检测到异常。</p>")

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="多态异常检测命令行接口",
        epilog="用法示例:\n"
               "  python muti_contrast.py --test test.csv\n"
               "  python muti_contrast.py --normal normal.csv --abnormal abnormal.csv --test test.csv\n"
               "  python muti_contrast.py --normal normal.csv --test test.csv\n"
               "  python muti_contrast.py --test test.csv --test_label test_label.csv\n"
               "  python muti_contrast.py --abnormal abnormal.csv --abnormal_only\n",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--normal", type=str, help="正常样本文件路径")
    parser.add_argument("--abnormal", type=str, help="异常样本文件路径")
    parser.add_argument("--test", type=str, help="待测数据文件路径")
    parser.add_argument("--test_label", type=str, help="测试集标签文件路径")
    parser.add_argument("--abnormal_only", action='store_true', help="只输入异常样本训练")
    parser.add_argument("--model_dir", type=str, default=None, help="指定模型保存和加载的目录（默认用当前输出目录/saved_models）")
    return parser.parse_args()

# --------- 主流程入口 ---------
def main():
    # 1. 解析参数与初始化输出目录
    args = parse_args()
    dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"output_{dt_str}_{str(uuid.uuid4())[:8]}"
    os.makedirs(out_dir, exist_ok=True)
    if args.model_dir is not None:
        model_dir = args.model_dir
        os.makedirs(model_dir, exist_ok=True)
    else:
        model_dir = os.path.join(out_dir, "saved_models")
        os.makedirs(model_dir, exist_ok=True)
    error_csv_path = os.path.join(out_dir, error_csv)
    train_logger = HtmlLogger(os.path.join(out_dir, "train.html"), title="训练集&测试集评估结果")
    retrain_logger = HtmlLogger(os.path.join(out_dir, "retrain.html"), title="伪标签反哺再训练结果")
    detect_logger = HtmlLogger(os.path.join(out_dir, "detect.html"), title="待测数据检测结果")

    # 2. 数据准备
    if not (args.normal or args.abnormal or args.test):
        err_msg = (
            "\n[错误] 未指定任何输入数据文件！\n"
            "请至少通过 --normal/--abnormal/--test 指定一个输入文件路径。\n"
        )
        train_logger.write(f"<pre>{err_msg}</pre>")
        train_logger.close()
        retrain_logger.close()
        detect_logger.close()
        exit(1)
    X, y_true, df_merged, df_normal, df_abnormal, df_test = prepare_data(args)

    # 3. 特征标准化与对比学习特征提取
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scaler = get_scaler(X, scaler_path)
    X_scaled = scaler.transform(X)
    if use_contrastive:
        encoder_path = os.path.join(model_dir, "encoder.pth")
        encoder = get_encoder(X_scaled, encoder_path, feature_dim=32, out_dir=out_dir, logger=train_logger)
        with torch.no_grad():
            X_feat = encoder(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    else:
        X_feat = X_scaled
    feat_dim = X_feat.shape[1]

    # 4. 是否自定义retrain伪标签训练
    train_logger.write("<h2>训练集&测试集评估结果</h2>")
    choice = input("是否需要自定义文件retrain? (y/n): ")
    if choice.lower() == 'y':
        files_str = input("请输入自定义retrain文件路径（可用英文逗号分隔多个文件）: ")
        file_list = [f.strip() for f in files_str.split(',') if f.strip()]
        X_pseudo_list = []
        y_pseudo_list = []
        for custom_file in file_list:
            if os.path.exists(custom_file):
                df_custom = read_table(custom_file)
                if df_custom is not None:
                    df_custom = preprocess_df(df_custom)
                    df_custom = add_stat_features(df_custom)
                    df_custom = df_custom.fillna(0)
                    missing_cols = [col for col in df_merged.columns if col not in df_custom.columns]
                    for col in missing_cols:
                        df_custom[col] = 0
                    df_custom = df_custom[df_merged.columns]
                    if 'label' in df_custom.columns:
                        y_custom = df_custom['label'].values.astype(int)
                        X_custom = df_custom.drop(columns=['label']).values
                    else:
                        label_type = input(f"请指定文件 {custom_file} 的标签（0-正常，1-异常）: ")
                        label_type = int(label_type) if label_type in ['0', '1'] else 0
                        X_custom = df_custom.values
                        y_custom = np.full(len(X_custom), label_type, dtype=int)
                    X_pseudo_list.append(X_custom)
                    y_pseudo_list.append(y_custom)
                else:
                    print(f"文件 {custom_file} 无法读取，跳过。")
            else:
                print(f"文件 {custom_file} 不存在，跳过。")
        if X_pseudo_list:
            X_pseudo = np.vstack(X_pseudo_list)
            y_pseudo = np.concatenate(y_pseudo_list)
            X_pseudo_scaled = scaler.transform(X_pseudo)
            if use_contrastive:
                with torch.no_grad():
                    X_pseudo_feat = encoder(torch.tensor(X_pseudo_scaled, dtype=torch.float32)).numpy()
            else:
                X_pseudo_feat = X_pseudo_scaled
            for contamination in contaminations:
                evaluate_models(X_pseudo_feat, y_pseudo, algorithms, contamination, model_dir, feat_dim, retrain_models=True, logger=retrain_logger)
    else:
        print("没有有效的自定义retrain文件，跳过retrain。")

    # 5. 训练集/测试集评估与伪标签反哺
    for contamination in contaminations:
        train_logger.write(f"<h3>异常比例 contamination={contamination*100:.2f}% (动态阈值)</h3>")
        metrics, final_y_vote, high_confidence_idx, avg_score = fusion_vote(
            fit_predict_models(X_feat, y_true, algorithms, contamination, model_dir, feat_dim),
            contamination, y_true, require_all=True, dynamic_c=True)
        df_metrics = pd.DataFrame(metrics)
        train_logger.write_tabulate(df_metrics, title="各模型性能指标")
        anomaly_indices = np.where(final_y_vote == 1)[0]
        print_anomaly_table_html(df_merged, anomaly_indices, train_logger, scores=avg_score)

        if len(high_confidence_idx) > 0:
            choice = input("检测到高置信伪异常样本。您是否需要指定文件进行再训练? (y/n): ")
            if choice.lower() == '0':
                custom_retrain_file = input("请输入自定义再训练文件的路径: ")
                if os.path.exists(custom_retrain_file):
                    df_custom_retrain = read_table(custom_retrain_file)
                    if df_custom_retrain is not None:
                        df_custom_retrain = preprocess_df(df_custom_retrain)
                        df_custom_retrain = add_stat_features(df_custom_retrain)
                        df_custom_retrain = df_custom_retrain.fillna(0)
                        missing_cols = [col for col in df_merged.columns if col not in df_custom_retrain.columns]
                        for col in missing_cols:
                            df_custom_retrain[col] = 0
                        df_custom_retrain = df_custom_retrain[df_merged.columns]
                        X_custom_retrain = df_custom_retrain.values
                        X_pseudo, y_pseudo = X_custom_retrain, np.ones(len(X_custom_retrain), dtype=int)
                    else:
                        retrain_logger.write("<p>指定的文件无法读取。</p>")
                        X_pseudo, y_pseudo = retrain_with_pseudo_anomaly(X, df_merged, high_confidence_idx, logger=retrain_logger)
                else:
                    retrain_logger.write("<p>指定的文件路径不存在或无法读取。</p>")
                    X_pseudo, y_pseudo = retrain_with_pseudo_anomaly(X, df_merged, high_confidence_idx, logger=retrain_logger)
            else:
                X_pseudo, y_pseudo = retrain_with_pseudo_anomaly(X, df_merged, high_confidence_idx, logger=retrain_logger)

            if y_pseudo is not None:
                X_pseudo_scaled = scaler.transform(X_pseudo)
                if use_contrastive:
                    with torch.no_grad():
                        X_pseudo_feat = encoder(torch.tensor(X_pseudo_scaled, dtype=torch.float32)).numpy()
                else:
                    X_pseudo_feat = X_pseudo_scaled
                evaluate_models(X_pseudo_feat, y_pseudo, algorithms, contamination, model_dir, feat_dim, retrain_models=True, logger=retrain_logger)

    # 6. 关闭日志器
    train_logger.close()
    retrain_logger.close()

    # 7. 检测阶段
    if df_test is not None:
        detect_logger.write("<h2>待测数据检测</h2>")
        X_test_scaled = scaler.transform(df_test.values)
        if use_contrastive:
            with torch.no_grad():
                X_test_feat = encoder(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
        else:
            X_test_feat = X_test_scaled
        test_labels = None
        if hasattr(args, "test_label") and args.test_label and os.path.exists(args.test_label):
            df_label = pd.read_csv(args.test_label)
            test_labels = np.zeros(len(df_test), dtype=int)
            test_labels[:] = -1
            for _, row in df_label.iterrows():
                if 0 <= row['index'] < len(df_test):
                    test_labels[int(row['index'])] = int(row['label'])

        for contamination in contaminations:
            detect_logger.write(f"<h3>异常比例 contamination={contamination*100:.2f}% (动态阈值)</h3>")
            test_results = {}
            y_score_list = []
            for name, Algo in algorithms:
                model_path = os.path.join(
                    model_dir,
                    f"{name.replace(' ', '_')}_cont{contamination}_feat{X_test_feat.shape[1]}.pkl"
                )
                model = joblib.load(model_path)
                y_pred = model.predict(X_test_feat)
                y_score = model.decision_function(X_test_feat)
                test_results[name] = y_pred
                y_score_list.append(y_score)

            y_score_list = np.array(y_score_list)
            y_score_list_norm = []
            for i in range(y_score_list.shape[0]):
                s = y_score_list[i]
                if np.std(s) > 1e-6:
                    s_norm = (s - np.mean(s)) / np.std(s)
                else:
                    s_norm = s - np.mean(s)
                y_score_list_norm.append(s_norm)
            y_score_list_norm = np.array(y_score_list_norm)
            avg_score = y_score_list_norm.mean(axis=0)
            top_idx, real_cont = dynamic_contamination(avg_score, contamination_min, contamination_max)
            final_y_vote = np.zeros_like(avg_score, dtype=int)
            final_y_vote[top_idx] = 1
            anomaly_indices = np.where(final_y_vote == 1)[0]
            print_anomaly_table_html(df_test, anomaly_indices, detect_logger, scores=avg_score)
            if len(anomaly_indices) > 0:
                df_test.iloc[anomaly_indices].to_csv(error_csv_path, index=False, mode='a', header=not os.path.exists(error_csv_path))
                detect_logger.write(f"<p>已将本轮检测为异常的样本追加写入 {error_csv_path}</p>")
            if test_labels is not None and np.any(test_labels != -1):
                valid_idx = test_labels != -1
                if np.any(valid_idx):
                    true_label = test_labels[valid_idx]
                    pred_label = final_y_vote[valid_idx]
                    perf = compute_all_metrics(true_label, pred_label, avg_score[valid_idx])
                    df_perf = pd.DataFrame([perf])
                    detect_logger.write_tabulate(df_perf, title="融合投票(动态阈值)在有标签的测试集上的性能")
        detect_logger.close()

if __name__ == "__main__":
    main()
