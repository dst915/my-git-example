import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import warnings
import os
import pickle
from tqdm import tqdm
import concurrent.futures
from itertools import combinations
import datetime
import re
from abc import ABC, abstractmethod
import requests
import sqlalchemy
warnings.filterwarnings("ignore")

HISTORY_FILE = 'history.pkl'

# ================== 字段业务关系多进程+大模型分析 ==================

def single_pair_llm_relation(args):
    import torch  # 加在这里，保证每个子进程都能用torch
    a, b, comment_a, comment_b, lang = args
    # 进程内加载模型（每个进程单独加载，适合CPU并发）
    global tokenizer, model
    if 'tokenizer' not in globals():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        MODEL_NAME = 'Qwen/Qwen1.5-0.5B-Chat'
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()
    if lang == "zh":
        prompt = (
            f"字段A: {a}（{comment_a}），字段B: {b}（{comment_b}）。"
            "请用一句话描述这两个字段在业务上的关系，包括但不限于：大于、小于、相等、子集、相关、倍数、无关等。"
            "如果无法判断请回答“无法判断”。"
        )
    else:
        prompt = (
            f"Field A: {a} ({comment_a}), Field B: {b} ({comment_b}). "
            "Describe the business relationship between these two fields in one sentence, "
            "including but not limited to: greater than, less than, equal, subset, correlation, multiple, unrelated, etc. "
            "If you cannot determine, reply '无法判断'."
        )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7
        )
    raw_output = output[0][input_ids.shape[1]:]
    answer = tokenizer.decode(raw_output, skip_special_tokens=True)
    if not answer.strip():
        answer = "没输出"
    print(f"\n【大模型推理】字段A: {a}, 字段B: {b}\nPrompt: {prompt}\nAnswer: {answer.strip()}\n{'='*60}")
    return (a, b, answer.strip())

def analyze_llm_field_relations(
    df, ratio_threshold=0.8, exclude_keywords=['id', 'dt'], lang='zh', max_workers=4
):
    colnames = list(df.columns)
    def is_excluded(col):
        col_lower = col.lower()
        return any(kw in col_lower for kw in exclude_keywords)
    num_cols = [col for col in df.select_dtypes(include=[np.number]).columns if not is_excluded(col)]
    pairs = list(combinations(num_cols, 2))
    msg = ""
    if not pairs:
        return "<p>无可分析的字段对。</p>"

    # 多进程推理
    args_list = [(a, b, "", "", lang) for a, b in pairs]
    relation_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_pair_llm_relation, args) for args in args_list]
        for idx, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="大模型业务关系推理")):
            a, b, rel = future.result()
            if rel and ("无法判断" not in rel and "Cannot determine" not in rel):
                mask = ~df[a].isna() & ~df[b].isna()
                n_valid = mask.sum()
                if n_valid == 0:
                    continue
                arr_a = df.loc[mask, a].values
                arr_b = df.loc[mask, b].values
                # 多种关系数学验证
                ratio_gt = (arr_a > arr_b).mean()
                ratio_lt = (arr_a < arr_b).mean()
                ratio_eq = (arr_a == arr_b).mean()
                ratio_subset = np.isin(arr_a, arr_b).mean()
                ratio_multiple = ((arr_a % arr_b == 0) & (arr_b != 0)).mean() if np.any(arr_b != 0) else 0
                corr = np.corrcoef(arr_a, arr_b)[0,1] if len(arr_a) > 1 else 0
                detected = []
                if ratio_gt >= ratio_threshold:
                    detected.append("A大于B")
                if ratio_lt >= ratio_threshold:
                    detected.append("A小于B")
                if ratio_eq >= ratio_threshold:
                    detected.append("A等于B")
                if ratio_subset >= ratio_threshold:
                    detected.append("A是B的子集")
                if ratio_multiple >= ratio_threshold:
                    detected.append("A是B的整数倍")
                if abs(corr) > 0.7:
                    detected.append(f"A和B高度相关(相关系数{corr:.2f})")
                relation_results.append((a, b, rel, detected, n_valid))

    # 生成HTML
    if relation_results:
        msg += "<h3>【大模型推理的字段业务关系及数学验证】</h3><ul>"
        for a, b, rel, detected, n_valid in relation_results:
            detected_str = ", ".join(detected) if detected else "无显著数学关系"
            msg += f"<li><b>{a}, {b}</b>：{rel} <br>（数学关系：{detected_str}，样本数: {n_valid}）</li>"
        msg += "</ul>"
    else:
        msg += "<p>无大模型可推理出明确业务关系的字段对。</p>"
    return msg

# ================== 数据加载多态接口（无改动） ==================
class DataLoader(ABC):
    @abstractmethod
    def load(self):
        pass
class CSVDataLoader(DataLoader):
    def __init__(self, filename):
        self.filename = filename
    def load(self):
        return pd.read_csv(self.filename, on_bad_lines='skip')
class ExcelDataLoader(DataLoader):
    def __init__(self, filename):
        self.filename = filename
    def load(self):
        return pd.read_excel(self.filename)
class DataFrameLoader(DataLoader):
    def __init__(self, df):
        self.df = df
    def load(self):
        return self.df
class StreamDataLoader(DataLoader):
    def __init__(self, stream):
        self.stream = stream
    def load(self):
        return pd.DataFrame(list(self.stream))
class DatabaseDataLoader(DataLoader):
    def __init__(self, conn_str, query):
        self.conn_str = conn_str
        self.query = query
    def load(self):
        engine = sqlalchemy.create_engine(self.conn_str)
        with engine.connect() as conn:
            df = pd.read_sql(self.query, conn)
        return df
class APIStreamDataLoader(DataLoader):
    def __init__(self, url, params=None, headers=None, json_path=None):
        self.url = url
        self.params = params or {}
        self.headers = headers or {}
        self.json_path = json_path
    def load(self):
        resp = requests.get(self.url, params=self.params, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        if self.json_path:
            for k in self.json_path.split('.'):
                data = data[k]
        return pd.DataFrame(data)
def get_loader(source):
    if isinstance(source, pd.DataFrame):
        return DataFrameLoader(source)
    if isinstance(source, str):
        if source.lower().endswith('.csv'):
            return CSVDataLoader(source)
        elif source.lower().endswith(('.xls', '.xlsx')):
            return ExcelDataLoader(source)
        elif source.lower().startswith('sqlite:///') or source.lower().startswith('mysql'):
            if '|' not in source:
                raise ValueError("数据库模式请用 conn_str|select_sql 格式")
            conn_str, query = source.split('|', 1)
            return DatabaseDataLoader(conn_str, query)
        elif source.lower().startswith('http'):
            url, *rest = source.split('?json_path=')
            json_path = rest[0] if rest else None
            return APIStreamDataLoader(url, json_path=json_path)
        else:
            raise ValueError("不支持的文件类型")
    if hasattr(source, '__iter__') and not isinstance(source, (str, bytes)):
        return StreamDataLoader(source)
    if isinstance(source, tuple) and len(source) == 2 and source[0].startswith('db'):
        return DatabaseDataLoader(source[0], source[1])
    if isinstance(source, tuple) and len(source) == 2 and source[0].startswith('http'):
        return APIStreamDataLoader(source[0], json_path=source[1])
    raise ValueError("未知的数据源类型")

# ================== 其它公式发现及HTML渲染函数（无改动） ==================

def canonicalize_formula(res):
    typ = res['类型']
    formula = res['公式']
    formula = formula.replace(' ', '')
    if typ in ['比率', '/', '乘法', '*']:
        m = re.match(r'(.+?)=(.+?)[/|*](.+)', formula)
        if m:
            left = m.group(1)
            op1 = m.group(2)
            op2 = m.group(3)
            fields = tuple(sorted([left, op1, op2]))
            return (typ, fields)
    if typ in ['加减混合', '+', '-']:
        m = re.match(r'(.+?)=(.+)', formula)
        if m:
            left = m.group(1)
            expr = m.group(2)
            terms = re.findall(r'([+-])?([A-Za-z0-9_]+)', expr)
            field_sign = {}
            for sign, field in terms:
                if not sign:
                    sign = '+'
                field_sign[field] = field_sign.get(field, 0) + (1 if sign == '+' else -1)
            fs_tuple = tuple(sorted(field_sign.items()))
            key = (typ, left, fs_tuple)
            return key
    return (typ, formula)

def deduplicate_results(group_results):
    seen = set()
    deduped = []
    for group in group_results:
        for res in group:
            key = canonicalize_formula(res)
            if key not in seen:
                seen.add(key)
                deduped.append(res)
    return deduped

def preprocess_dataframe(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def all_subgroups(group, min_group_size=2):
    subgroups = []
    for l in range(min_group_size, len(group)+1):
        for sub in combinations(group, l):
            subgroups.append(list(sub))
    return subgroups

def smart_field_groups(fields, field_comments, df, min_group_size=2, max_group_size=6):
    num_fields = [f for f in fields if pd.api.types.is_numeric_dtype(df[f]) or is_id_field(f)]
    groups = []
    for l in range(min_group_size, min(max_group_size, len(num_fields)) + 1):
        for group in combinations(num_fields, l):
            groups.append(list(group))
    return groups

def is_id_field(field):
    field_lower = field.lower()
    return 'id' in field_lower and not any(x in field_lower for x in ['sum', 'amount', 'fee', 'code', 'no', 'token', 'status', 'time'])

def get_valid_amount_fields(fields, df, zero_ratio_thresh=0.95):
    valid_fields = []
    for f in fields:
        if not pd.api.types.is_numeric_dtype(df[f]):
            continue
        arr = df[f].dropna().values.astype(float)
        if len(arr) == 0:
            continue
        if np.all(arr == 0) or np.var(arr) < 1e-6:
            continue
        zero_ratio = np.sum(arr == 0) / len(arr)
        if zero_ratio > zero_ratio_thresh:
            continue
        valid_fields.append(f)
    return valid_fields

def risk_report(y_true, y_pred, mask, max_show=5):
    risks = []
    for idx, (ok, yt, yp) in enumerate(zip(mask, y_true, y_pred)):
        if not ok:
            if isinstance(yt, str) or isinstance(yp, str):
                diff = f"{yt} != {yp}"
            else:
                diff = yt - yp
            risks.append({'index': idx, 'true': yt, 'pred': yp, 'diff': diff})
    return risks[:max_show]

def pretty_formula(target, others, op='+'):
    if op == '+':
        return f"{target} = {' + '.join(others)}"
    elif op == '-':
        return f"{target} = {' - '.join(others)}"
    elif op == '/':
        return f"{target} = {' / '.join(others)}"
    elif op == '*':
        return f"{target} = {' * '.join(others)}"
    elif op == 'rate':
        return f"{target} = {others[0]} / {others[1]}"
    elif op == 'growth':
        return f"{target} = ({others[0]} - {others[1]}) / {others[1]}"
    elif op == 'equal':
        return f"{' = '.join([target] + others)}"
    elif op == 'mix':
        fields, weights = others
        terms = []
        for f, w in zip(fields, weights):
            if w == 1:
                terms.append(f"+{f}")
            elif w == -1:
                terms.append(f"-{f}")
        expr = ' '.join(terms).replace('+-', '- ')
        expr = expr.lstrip('+').strip()
        return f"{target} = {expr}"
    return f"{target} 与 {others} 存在关系"

def verify_business_relations(df, group, ratio_thresh=0.85, corr_thresh=0.5, r2_thresh=0.5, discover_limit=200):
    results = []
    id_fields = [f for f in group if is_id_field(f)]
    if id_fields:
        id_field = id_fields[0]
        arr_id = df[id_field].astype(str).fillna('')
        min_len = min(len(df[f]) for f in group)
        arr_id = arr_id[:min_len]
        for f in group:
            if f == id_field:
                continue
            arr_f = df[f].astype(str).fillna('')[:min_len]
            mask = (arr_id == arr_f)
            ratio = mask.sum() / min_len
            if ratio < 0.9:
                continue
            results.append({
                '类型': '相等',
                '公式': pretty_formula(id_field, [f], 'equal'),
                '满足比例': float(ratio),
                '相关系数': float('nan'),
                'R2': float('nan'),
                'risk': risk_report(arr_id, arr_f, mask)
            })
        return results

    static_fields = [f for f in group if df[f].nunique() <= 1]
    valid_fields = [f for f in group if f not in static_fields]
    if len(valid_fields) < 2:
        return results
    df_discover = df.head(discover_limit)
    num_cols = get_valid_amount_fields(valid_fields, df_discover)

    # 加减混合
    for target in num_cols:
        others = [col for col in num_cols if col != target]
        if len(others) < 2:
            continue
        cols = [target] + others
        df_valid = df_discover[cols].dropna(axis=0, how='any')
        if len(df_valid) < 2:
            continue
        y2 = df_valid[target].values.astype(float)
        arrs = df_valid[others].values.astype(float)
        reg = LinearRegression(fit_intercept=False).fit(arrs, y2)
        weights = reg.coef_
        weights_rounded = []
        valid = True
        for w in weights:
            if np.isclose(w, 1, atol=1e-2):
                weights_rounded.append(1)
            elif np.isclose(w, -1, atol=1e-2):
                weights_rounded.append(-1)
            elif np.isclose(w, 0, atol=1e-2):
                weights_rounded.append(0)
            else:
                valid = False
                break
        if not valid or np.count_nonzero(weights_rounded) < 2:
            continue
        filtered_others = [c for c, w in zip(others, weights_rounded) if w != 0]
        filtered_weights = [w for w in weights_rounded if w != 0]
        arrs_filtered = arrs[:, np.array(weights_rounded) != 0]
        mixed_sum = arrs_filtered.dot(filtered_weights)
        mask = np.isclose(y2, mixed_sum, rtol=1e-3)
        ratio = mask.sum() / len(y2)
        try:
            corr2 = pearsonr(y2, mixed_sum)[0]
            r2_2 = LinearRegression(fit_intercept=False).fit(arrs_filtered, y2).score(arrs_filtered, y2)
        except Exception:
            corr2 = np.nan
            r2_2 = np.nan
        if (
            ratio >= ratio_thresh and
            (corr2 is not None and not np.isnan(corr2) and corr2 >= corr_thresh) and
            (r2_2 is not None and not np.isnan(r2_2) and r2_2 >= r2_thresh)
        ):
            df_all_valid = df[[target]+filtered_others].dropna(axis=0, how='any')
            if len(df_all_valid) < 2:
                continue
            y2_all = df_all_valid[target].values.astype(float)
            arrs_all = df_all_valid[filtered_others].values.astype(float)
            mixed_sum_all = arrs_all.dot(filtered_weights)
            mask_all = np.isclose(y2_all, mixed_sum_all, rtol=1e-1)
            ratio_all = mask_all.sum() / len(y2_all)
            try:
                corr2_all = pearsonr(y2_all, mixed_sum_all)[0]
                r2_2_all = LinearRegression(fit_intercept=False).fit(arrs_all, y2_all).score(arrs_all, y2_all)
            except Exception:
                corr2_all = np.nan
                r2_2_all = np.nan
            results.append({
                '类型': '加减混合',
                '公式': pretty_formula(target, [filtered_others, filtered_weights], 'mix'),
                '满足比例': float(ratio_all),
                '相关系数': float(corr2_all),
                'R2': float(r2_2_all),
                'risk': risk_report(y2_all, mixed_sum_all, mask_all)
            })
    # 比率
    for target in num_cols:
        for a in num_cols:
            for b in num_cols:
                if len({target, a, b}) < 3:
                    continue
                if ('sum' in target.lower() and (('sum' in a.lower()) or ('sum' in b.lower()))) or \
                   (not ('sum' in target.lower()) and (('sum' in a.lower()) or ('sum' in b.lower()))):
                    continue
                y = df_discover[target].dropna().values.astype(float)
                x = df_discover[a].dropna().values.astype(float)
                z = df_discover[b].dropna().values.astype(float)
                min_len = min(len(y), len(x), len(z))
                if min_len == 0 or np.any(z[:min_len] == 0):
                    continue
                y2 = y[:min_len]
                mask = np.isclose(y2, x[:min_len] / z[:min_len], rtol=1e-3)
                ratio = mask.sum() / min_len
                try:
                    corr2 = pearsonr(y2, x[:min_len] / z[:min_len])[0]
                    r2_2 = LinearRegression().fit(
                        (x[:min_len] / z[:min_len]).reshape(-1, 1), y2
                    ).score((x[:min_len] / z[:min_len]).reshape(-1, 1), y2)
                except Exception:
                    corr2 = np.nan
                    r2_2 = np.nan
                if (
                    ratio >= ratio_thresh and
                    (corr2 is not None and not np.isnan(corr2) and corr2 >= corr_thresh) and
                    (r2_2 is not None and not np.isnan(r2_2) and r2_2 >= r2_thresh)
                ):
                    y_all = df[target].dropna().values.astype(float)
                    x_all = df[a].dropna().values.astype(float)
                    z_all = df[b].dropna().values.astype(float)
                    min_len_all = min(len(y_all), len(x_all), len(z_all))
                    if min_len_all == 0 or np.any(z_all[:min_len_all] == 0):
                        continue
                    y2_all = y_all[:min_len_all]
                    mask_all = np.isclose(y2_all, x_all[:min_len_all] / z_all[:min_len_all], rtol=1e-1)
                    ratio_all = mask_all.sum() / min_len_all
                    try:
                        corr2_all = pearsonr(y2_all, x_all[:min_len_all] / z_all[:min_len_all])[0]
                        r2_2_all = LinearRegression().fit(
                            (x_all[:min_len_all] / z_all[:min_len_all]).reshape(-1, 1), y2_all
                        ).score((x_all[:min_len_all] / z_all[:min_len_all]).reshape(-1, 1), y2_all)
                    except Exception:
                        corr2_all = np.nan
                        r2_2_all = np.nan
                    results.append({
                        '类型': '比率',
                        '公式': pretty_formula(target, [a, b], 'rate'),
                        '满足比例': float(ratio_all),
                        '相关系数': float(corr2_all),
                        'R2': float(r2_2_all),
                        'risk': risk_report(y2_all, x_all[:min_len_all] / z_all[:min_len_all], mask_all)
                    })
    # 增长率
    for target in num_cols:
        for a in num_cols:
            if target == a:
                continue
            if ('sum' in target.lower() and 'sum' in a.lower()) or \
               (not ('sum' in target.lower()) and 'sum' in a.lower()):
                continue
            y = df_discover[target].dropna().values.astype(float)
            x = df_discover[a].dropna().values.astype(float)
            if len(y) != len(x) or len(y) < 2:
                continue
            prev_x = np.roll(x, 1)
            prev_x[0] = np.nan
            mask = np.isclose(y, (x/prev_x - 1), rtol=1e-3, equal_nan=True)
            ratio = np.nansum(mask) / np.sum(~np.isnan(mask))
            try:
                growth = x/prev_x - 1
                corr2 = pearsonr(y[~np.isnan(growth)], growth[~np.isnan(growth)])[0]
                r2_2 = LinearRegression().fit(
                    growth[~np.isnan(growth)].reshape(-1, 1), y[~np.isnan(growth)]
                ).score(growth[~np.isnan(growth)].reshape(-1, 1), y[~np.isnan(growth)])
            except Exception:
                corr2 = np.nan
                r2_2 = np.nan
            if (
                ratio >= ratio_thresh and
                (corr2 is not None and not np.isnan(corr2) and corr2 >= corr_thresh) and
                (r2_2 is not None and not np.isnan(r2_2) and r2_2 >= r2_thresh)
            ):
                y_all = df[target].dropna().values.astype(float)
                x_all = df[a].dropna().values.astype(float)
                if len(y_all) != len(x_all) or len(y_all) < 2:
                    continue
                prev_x_all = np.roll(x_all, 1)
                prev_x_all[0] = np.nan
                growth_all = x_all / prev_x_all - 1
                mask_all = np.isclose(y_all, growth_all, rtol=1e-1, equal_nan=True)
                ratio_all = np.nansum(mask_all) / np.sum(~np.isnan(mask_all))
                try:
                    corr2_all = pearsonr(y_all[~np.isnan(growth_all)], growth_all[~np.isnan(growth_all)])[0]
                    r2_2_all = LinearRegression().fit(
                        growth_all[~np.isnan(growth_all)].reshape(-1, 1), y_all[~np.isnan(growth_all)]
                    ).score(growth_all[~np.isnan(growth_all)].reshape(-1, 1), y_all[~np.isnan(growth_all)])
                except Exception:
                    corr2_all = np.nan
                    r2_2_all = np.nan
                results.append({
                    '类型': '增长率',
                    '公式': pretty_formula(target, [a, '上期'], 'growth'),
                    '满足比例': float(ratio_all),
                    '相关系数': float(corr2_all),
                    'R2': float(r2_2_all),
                    'risk': risk_report(y_all, growth_all, mask_all)
                })
    return results

def compare_math_relations(new_results, old_results):
    def rel_key(res):
        return (res['类型'], res['公式'])
    new_set = set([rel_key(r) for r in new_results])
    old_set = set([rel_key(r) for r in old_results])
    added = new_set - old_set
    removed = old_set - new_set
    return added, removed

def flatten_results(all_group_results):
    flat = []
    for group_results in all_group_results:
        flat.extend(group_results)
    return flat

def html_escape(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')

def render_formula_row(res):
    risk_str = ''
    if res.get('risk'):
        risk_str = '<br>'.join(
            f"[风险] 行号: {r['index']}, 真实值: {r['true']}, 预测值: {r['pred']}, 差值: {r['diff']}"
            for r in res['risk']
        )
    return f"""
    <tr>
        <td>{html_escape(res['类型'])}</td>
        <td>{html_escape(res['公式'])}</td>
        <td>{html_escape(res['满足比例'])}</td>
        <td>{risk_str}</td>
    </tr>
    """

def render_html(title, group_results, added=None, removed=None, similar_html=None):
    date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    table_rows = []
    for res in deduplicate_results(group_results):
        table_rows.append(render_formula_row(res))

    table_html = f"""
    <table>
        <thead>
            <tr>
                <th>类型</th>
                <th>公式</th>
                <th>满足比例</th>
                <th>风险报告</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
    """

    added_html = ""
    if added:
        added_html = "<div class='added'><h3>本次新增的数学规律：</h3><ul>" + \
            "".join(f"<li>{html_escape(str(rel))}</li>" for rel in added) + "</ul></div>"
    removed_html = ""
    if removed:
        removed_html = "<div class='removed'><h3>本次不符合之前的数学规律：</h3><ul>" + \
            "".join(f"<li>{html_escape(str(rel))}</li>" for rel in removed) + "</ul></div>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #222; padding: 30px; }}
            h1 {{ color: #3b5998; }}
            table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 2px 6px #eee; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background: #f0f2f5; }}
            tr:nth-child(even){{background-color: #f9fafb;}}
            .added {{ color: #388e3c; margin: 20px 0; }}
            .removed {{ color: #d32f2f; margin: 20px 0; }}
            .date {{ color: #888; font-size: 0.95em; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="date">分析时间：{date_str}</div>
        {similar_html if similar_html else ""}
        {added_html}
        {removed_html}
        {table_html}
    </body>
    </html>
    """
    return html_content
# ================== 主流程 ==================
if __name__ == '__main__':
    print("AI智能字段组合与数学关系自动发现 (HTML输出版)")
    last_results = None
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'rb') as f:
            last_results = pickle.load(f)
    seen_combinations = set()  # 组合去重集合
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MODEL_NAME = 'Qwen/Qwen1.5-0.5B-Chat'
    print("正在预下载模型，请耐心等待...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()
    print("模型下载完成！")
    del tokenizer, model

    print("正在加载本地小型大模型用于字段业务关系推理（Qwen1.5-0.5B-Chat）...")

    while True:
        print("请输入数据源（csv/excel文件名，或数据库/接口，或exit退出，或输入'demo'体验内存流）：")
        print("数据库示例：sqlite:///test.db|select * from table")
        print("API接口示例：https://api.xxx.com/data?json_path=data.items")
        source = input().strip()
        if source.lower() == 'exit':
            print("程序已退出。")
            break
        if source.lower() == 'demo':
            demo_data = [
                {"A": 1, "B": 2, "C": 3},
                {"A": 2, "B": 3, "C": 5},
                {"A": 3, "B": 4, "C": 7},
                {"A": 4, "B": 5, "C": 9},
            ]
            loader = get_loader(demo_data)
        else:
            if not os.path.exists(source):
                print("文件不存在，请重新输入。")
                continue
            try:
                loader = get_loader(source)
            except Exception as e:
                print(f"数据源类型不支持: {e}")
                continue

        try:
            df = loader.load()
        except Exception as e:
            print(f"加载数据失败: {e}")
            continue

        df = preprocess_dataframe(df)
        # 字段业务关系+多种数学关系分析
        similar_col_html = analyze_llm_field_relations(df, ratio_threshold=0.8, max_workers=4)
        fields = list(df.columns)
        field_comments = [''] * len(fields)
        groups = smart_field_groups(fields, field_comments, df, min_group_size=2, max_group_size=6)
        print("AI智能筛选出的高概率字段组合，正在分析...")

        all_group_results = []
        total = sum(len(all_subgroups(group, min_group_size=2)) for group in groups)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for group in groups:
                subgroups = all_subgroups(group, min_group_size=2)
                for subgroup in subgroups:
                    key = frozenset(subgroup)
                    if key in seen_combinations:
                        continue
                    seen_combinations.add(key)
                    futures.append(executor.submit(
                        verify_business_relations, df, subgroup, 0.85, 0.5, 0.6, 200
                    ))

            bar = tqdm(total=len(futures), ncols=80, desc="公式发现中")
            for future in concurrent.futures.as_completed(futures):
                busi_results = future.result()
                all_group_results.append(busi_results)
                bar.update(1)
            bar.close()

        flat_results = flatten_results(all_group_results)
        added, removed = None, None
        if last_results is not None:
            added, removed = compare_math_relations(flat_results, last_results)
        with open(HISTORY_FILE, 'wb') as f:
            pickle.dump(flat_results, f)
        last_results = flat_results

        html_content = render_html(
            title="AI智能字段组合与数学规则自动发现报告",
            group_results=all_group_results,
            added=added if added else None,
            removed=removed if removed else None,
            similar_html=similar_col_html
        )
        out_html = f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"本次分析已完成，结果已保存为 {out_html}，可继续输入下一个数据源或输入exit退出。")
