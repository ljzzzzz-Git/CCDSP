# -*- coding: utf-8 -*-
"""
结直肠癌药物敏感性预测系统 - Flask后端服务
核心功能：
1. 文件上传与格式验证（3个CSV文件：突变、药物SMILE、ssGSEA富集）
2. 模型预测任务进度跟踪（多线程安全）
3. 预测结果可视化数据解析（支持Top10药物统计）
4. 参考数据/预测结果下载
5. 系统使用指南与图文解答
"""
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import uuid
from datetime import datetime
import predict_model  # 模型预测核心模块
import threading
import time
import pandas as pd

# ====================== 全局配置区 ======================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'          
app.config['RESULT_FOLDER'] = 'results'          
app.config['REFERENCE_FOLDER'] = 'static/data'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  
app.config['SECRET_KEY'] = 'colorectal_cancer_pred_2026'  

# 初始化目录
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], app.config['REFERENCE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 全局任务进度管理
task_progress = {}          
progress_lock = threading.Lock()  

# ====================== 工具函数区 ======================
def is_csv(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def update_progress(task_id: str, step: int, percentage: int, message: str) -> None:
    """线程安全更新任务进度"""
    with progress_lock:
        task_progress[task_id] = {
            'step': step,
            'percentage': percentage,
            'message': message,
            'timestamp': time.time()
        }

def clear_expired_tasks() -> None:
    """后台线程清理过期任务（30分钟）"""
    while True:
        with progress_lock:
            current_time = time.time()
            expired_task_ids = [
                tid for tid, info in task_progress.items()
                if current_time - info['timestamp'] > 1800
            ]
            for tid in expired_task_ids:
                del task_progress[tid]
        time.sleep(60)

# 启动过期任务清理线程
threading.Thread(target=clear_expired_tasks, daemon=True).start()

# ====================== 路由处理区 ======================
@app.route('/')
def home() -> str:
    """系统主页路由"""
    return render_template('index.html')

# ====================== 使用指南 ========================
@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/reference')
def reference() -> str:
    """参考数据页面路由"""
    ref_files = [
        f for f in os.listdir(app.config['REFERENCE_FOLDER'])
        if f.endswith('.csv') and not f.startswith('.')
    ]
    return render_template('reference.html', ref_files=ref_files)

@app.route('/download_reference/<filename>')
def download_reference(filename: str) -> tuple:
    """参考数据下载路由"""
    ref_file_path = os.path.join(app.config['REFERENCE_FOLDER'], filename)
    if not os.path.exists(ref_file_path):
        return jsonify({"status": "error", "msg": "参考文件不存在"}), 404
    return send_file(
        ref_file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv; charset=utf-8-sig'
    )

@app.route('/predict')
def predict_page() -> str:
    """预测页面路由"""
    task_id = str(uuid.uuid4())[:8]
    session['task_id'] = task_id
    update_progress(task_id, 0, 0, "等待文件上传...")
    return render_template('predict.html', task_id=task_id)

@app.route('/result/<filename>')
def result_page(filename: str) -> tuple:
    """预测结果可视化路由：Top10药物统计（放宽过滤条件）"""
    result_file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(result_file_path):
        return "结果文件不存在", 404
    
    # 读取CSV文件
    try:
        result_df = pd.read_csv(result_file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        result_df = pd.read_csv(result_file_path, encoding='gbk')
    except Exception:
        empty_stats = {'total':0, 'high':0, 'low':0, 'drug_stats_top10':{}}
        return render_template('result.html', filename=filename, stats=empty_stats)
    
    # 列名校验与兼容
    required_columns = ['预测敏感性', 'drug_name']
    for target_col in required_columns:
        if target_col not in result_df.columns:
            similar_cols = [
                col for col in result_df.columns
                if target_col in col or col.lower() == target_col.lower()
            ]
            if similar_cols:
                result_df.rename(columns={similar_cols[0]: target_col}, inplace=True)
            else:
                empty_stats = {
                    'total': len(result_df),
                    'high': 0,
                    'low': 0,
                    'drug_stats_top10': {}
                }
                return render_template('result.html', filename=filename, stats=empty_stats)
    
    # 数据清洗（仅过滤高/低敏感）
    valid_df = result_df[result_df['预测敏感性'].isin(['高', '低'])]
    total_valid_pairs = len(valid_df)
    high_sensitivity_count = len(valid_df[valid_df['预测敏感性'] == '高'])
    low_sensitivity_count = len(valid_df[valid_df['预测敏感性'] == '低'])
    
    # Top10药物统计（放宽过滤条件）
    drug_stats_top10 = {}
    valid_drug_df = valid_df[
        valid_df['drug_name'].notna() & 
        (valid_df['drug_name'] != '') &
        (valid_df['drug_name'].str.strip() != '')
    ]
    
    if len(valid_drug_df) > 0:
        # 统计每个药物的总数量
        drug_total_count = valid_drug_df.groupby('drug_name').size().reset_index(name='total_count')
        # 统计每个药物的高敏感数量
        drug_high_count = valid_drug_df[valid_df['预测敏感性'] == '高'].groupby('drug_name').size().reset_index(name='high_count')
        # 合并数据并计算占比
        drug_stats = pd.merge(drug_total_count, drug_high_count, on='drug_name', how='left').fillna(0)
        drug_stats['high_ratio'] = round((drug_stats['high_count'] / drug_stats['total_count']) * 100, 2)
        # 按总数量降序排序，取前10
        drug_stats_sorted = drug_stats.sort_values(by='total_count', ascending=False).head(10)
        # 封装Top10数据
        for _, row in drug_stats_sorted.iterrows():
            drug_stats_top10[row['drug_name']] = {
                'total_count': int(row['total_count']),
                'high_count': int(row['high_count']),
                'high_ratio': row['high_ratio']
            }
    
    # 封装最终统计数据
    visualization_stats = {
        'total': total_valid_pairs,
        'high': high_sensitivity_count,
        'low': low_sensitivity_count,
        'drug_stats_top10': drug_stats_top10
    }
    
    return render_template('result.html', filename=filename, stats=visualization_stats)

@app.route('/get_progress/<task_id>')
def get_progress(task_id: str) -> str:
    """获取任务进度路由"""
    with progress_lock:
        progress_info = task_progress.get(task_id, {
            'step': 0,
            'percentage': 0,
            'message': "任务未开始或已超时"
        })
    return jsonify(progress_info)

@app.route('/upload_predict', methods=['POST'])
def upload_predict() -> str:
    task_id = request.form.get('task_id') or session.get('task_id')
    if not task_id:
        return jsonify({"status": "error", "msg": "任务ID缺失，请刷新页面重试"})
    
    try:
        update_progress(task_id, 1, 20, "正在接收上传文件...")
        mut_file = request.files.get('mut_csv')
        smile_file = request.files.get('smile_csv')
        ssgsea_file = request.files.get('ssgsea_csv')
        
        if not all([mut_file, smile_file, ssgsea_file]):
            update_progress(task_id, 1, 0, "错误：缺少必需文件")
            return jsonify({"status": "error", "msg": "请上传所有3个CSV文件，不能为空"})
        
        if not all([is_csv(f.filename) for f in [mut_file, smile_file, ssgsea_file]]):
            update_progress(task_id, 1, 0, "错误：文件格式非CSV")
            return jsonify({"status": "error", "msg": "所有文件必须为CSV格式（.csv后缀）"})
        
        # 保存文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        short_uuid = str(uuid.uuid4())[:6]
        mut_save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"mut_{timestamp}_{short_uuid}_{mut_file.filename}")
        smile_save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"smile_{timestamp}_{short_uuid}_{smile_file.filename}")
        ssgsea_save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"ssgsea_{timestamp}_{short_uuid}_{ssgsea_file.filename}")
        mut_file.save(mut_save_path)
        smile_file.save(smile_save_path)
        ssgsea_file.save(ssgsea_save_path)
        
        # 验证CSV格式
        update_progress(task_id, 2, 30, "验证CSV文件格式...")
        try:
            mut_df = pd.read_csv(mut_save_path, encoding='utf-8-sig')
            smile_df = pd.read_csv(smile_save_path, encoding='utf-8-sig')
            ssgsea_df = pd.read_csv(ssgsea_save_path, encoding='utf-8-sig')
            
            if not all(col in mut_df.columns for col in ['cell_idx', 'cell_line']):
                raise ValueError("突变CSV缺少必需列：cell_idx、cell_line")
            if not all(col in smile_df.columns for col in ['drug_idx', 'drug_name']):
                raise ValueError("药物CSV缺少必需列：drug_idx、drug_name")
            if not all(col in ssgsea_df.columns for col in ['cell_idx', 'cell_name']):
                raise ValueError("富集CSV缺少必需列：cell_idx、cell_name")
            
            update_progress(task_id, 2, 40, "CSV文件格式验证通过")
        except Exception as e:
            error_msg = f"文件格式验证失败：{str(e)}"
            update_progress(task_id, 2, 0, f"错误：{error_msg}")
            return jsonify({"status": "error", "msg": error_msg})
        
        # 加载模型
        update_progress(task_id, 3, 50, "加载预测模型和标准化器...")
        try:
            # 模拟模型加载
            time.sleep(1)
            update_progress(task_id, 3, 60, "模型加载完成")
        except Exception as e:
            error_msg = f"模型加载失败：{str(e)}"
            update_progress(task_id, 3, 0, f"错误：{error_msg}")
            return jsonify({"status": "error", "msg": error_msg})
        
        # 执行预测
        update_progress(task_id, 4, 70, "生成药物-细胞配对数据...")
        result_filename = f"result_{timestamp}_{short_uuid}.csv"
        result_save_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        update_progress(task_id, 4, 80, "执行批量预测（耗时较长，请等待）...")
        
        # 调用你的模型
        predict_success = predict_model.predict_three_csv(
            mut_csv=mut_save_path,
            smile_csv=smile_save_path,
            ssgsea_csv=ssgsea_save_path,
            output_csv=result_save_path,
            task_id=task_id,
            update_progress_func=update_progress
        )
        
        # 返回结果
        if predict_success and os.path.exists(result_save_path):
            update_progress(task_id, 5, 100, "预测完成，准备结果展示...")
            result_df = pd.read_csv(result_save_path, encoding='utf-8-sig')
            result_row_count = len(result_df)
            return jsonify({
                "status": "success",
                "msg": f"预测完成！共生成{result_row_count}条药物敏感性结果",
                "result_url": f"/result/{result_filename}",
                "download_url": f"/download/{result_filename}",
                "result_count": result_row_count
            })
        else:
            update_progress(task_id, 4, 0, "错误：模型预测执行失败")
            return jsonify({"status": "error", "msg": "模型预测失败，请检查CSV数据格式或模型状态"})
    
    except Exception as e:
        error_msg = f"系统异常：{str(e)}"
        update_progress(task_id, 0, 0, f"错误：{error_msg}")
        return jsonify({"status": "error", "msg": error_msg})

@app.route('/download/<filename>')
def download(filename: str) -> tuple:
    """预测结果下载路由"""
    result_file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(result_file_path):
        return jsonify({"status": "error", "msg": "结果文件不存在"}), 404
    return send_file(
        result_file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv; charset=utf-8-sig'
    )

# ====================== 程序入口 ======================
if __name__ == '__main__':
    print("=== 结直肠癌药物敏感性预测系统启动 ===")
    print(f"服务地址：http://localhost:5000")
    print(f"调试模式：{'开启' if app.debug else '关闭'}")
    app.run(debug=True, host='0.0.0.0', port=5000)

'''
├── app_flask.py               # 后端主程序
├── predict_model.py           # 预测模型核心函数
├── templates/                 # HTML模板文件夹
│   ├── index.html             # 首页
│   ├── guide.html             # 系统使用指南
│   ├── reference.html         # 参考数据下载页
│   ├── result.html            # 预测结果可视化页
│   └── predict.html           # 药物预测上传页
└── static/                    # 静态资源文件夹
    ├── css/
    │   ├── style.css          # 通用样式
    │   └── index.css          # 主页专属样式
    └── data/                  # 参考CSV数据文件夹
'''