#!/usr/bin/env python3
"""
批量运行基因组预测任务
为番茄数据集的每个表型依次执行genomic_prediction_v2.py命令
使用预处理功能自动转换TXT表型文件为CSV格式
"""

import os
import subprocess
import time
from pathlib import Path
import glob

# 配置参数
GENO_FILE = 'geno/melon_geno.csv'  # 单个基因型文件
PHENO_DIR = 'melon_phenotyp_continuous'  # 表型文件目录
RESULTS_BASE_DIR = 'melon_multi_traits_results'
PREPROCESS_BASE_DIR = 'melon_processed_data'  # 预处理数据保存目录
SPECIES_NAME = 'melon'  # 固定物种名称

# genomic_prediction_v2.py 的参数配置
PREDICTION_PARAMS = {
    # 预处理相关参数
    'enable_preprocess': True,  # 启用预处理
    'task_type': 'regression',  # 任务类型
    'standardize_phenotype': True,  # 启用表型标准化
    
    # 模型训练参数
    'use_default_params': True,
    'use_stacking': True,
    'top_n_models': 3,
    'n_splits': 5,
    'n_repeats': 50,  # 修改为50
    'parallel_jobs': 50,  # 修改为50
    'test_size': 0.3,  # 修改为0.3
    'threads': 1,  # 修改为1
    'use_same_test_set': True,
    'save_representative': True,  # 修改为True
}

def get_pheno_files():
    """获取所有表型文件"""
    pheno_files = glob.glob(os.path.join(PHENO_DIR, '*.txt'))
    return sorted(pheno_files)

def parse_pheno_filename(pheno_file):
    """从表型文件名解析表型名称"""
    basename = os.path.basename(pheno_file)
    # 例如: Anthracnose.txt -> trait=Anthracnose
    trait = basename.replace('.txt', '')
    return SPECIES_NAME, trait

def build_command(geno_file, pheno_file, species, trait):
    """构建genomic_prediction_v2.py命令"""
    # 结果保存在统一文件夹下，每个表型一个子目录
    results_dir = os.path.join(RESULTS_BASE_DIR, trait)
    
    # 预处理输出前缀
    preprocess_prefix = os.path.join(PREPROCESS_BASE_DIR, trait)
    
    # 日志文件也保存在统一的日志目录下
    log_dir = os.path.join(RESULTS_BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{trait}.log")
    
    # 创建预处理目录
    os.makedirs(PREPROCESS_BASE_DIR, exist_ok=True)
    
    cmd = [
        'python', 'genomic_prediction_v2.py',
        '--enable_preprocess',  # 新增：启用预处理
        '--matrix_file', geno_file,  # 修改：使用matrix_file而不是geno_file
        '--raw_pheno_file', pheno_file,  # 修改：使用raw_pheno_file
        '--preprocess_prefix', preprocess_prefix,  # 新增：预处理输出前缀
        '--target_trait', trait,
        '--task_type', 'regression',  # 新增：明确指定任务类型
        '--results_dir', results_dir,
    ]
    
    # 添加其他参数
    for param, value in PREDICTION_PARAMS.items():
        # 跳过已经添加的参数
        if param in ['enable_preprocess', 'task_type']:
            continue
            
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{param}')
        else:
            cmd.extend([f'--{param}', str(value)])
    
    return cmd, log_file, results_dir

def run_task(geno_file, pheno_file, species, trait):
    """运行单个预测任务（同步执行，等待完成）"""
    cmd, log_file, results_dir = build_command(geno_file, pheno_file, species, trait)
    
    print(f"\n{'='*80}")
    print(f"物种: {species}, 表型: {trait}")
    print(f"基因型文件: {geno_file}")
    print(f"表型文件: {pheno_file}")
    print(f"结果目录: {results_dir}")
    print(f"日志文件: {log_file}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # 打开日志文件
    with open(log_file, 'w') as log_f:
        # 设置环境变量，确保能找到conda环境下的libstdc++.so.6
        env = os.environ.copy()
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            lib_path = os.path.join(conda_prefix, 'lib')
            current_ld = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld}"
        
        # 同步执行命令，等待完成
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        
        print(f">>> 任务开始执行，PID: {process.pid}")
        print(f">>> 实时日志: tail -f {log_file}")
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print(f"✓ 任务完成成功")
        else:
            print(f"✗ 任务失败，返回码: {return_code}")
    
    return log_file, results_dir, return_code

def main():
    """主函数"""
    print("开始批量基因组预测任务（串行执行，启用预处理功能）...")
    print(f"物种: {SPECIES_NAME}")
    print(f"基因型文件: {GENO_FILE}")
    print(f"表型文件目录: {PHENO_DIR}")
    print(f"预处理数据目录: {PREPROCESS_BASE_DIR}/")
    print(f"结果保存目录: {RESULTS_BASE_DIR}/")
    print()
    
    # 检查基因型文件是否存在
    if not os.path.exists(GENO_FILE):
        print(f"错误: 找不到基因型文件 {GENO_FILE}")
        return
    
    # 获取所有表型文件
    pheno_files = get_pheno_files()
    
    if not pheno_files:
        print(f"错误: 在 {PHENO_DIR} 目录中没有找到表型文件")
        return
    
    print(f"找到 {len(pheno_files)} 个表型文件")
    print()
    
    # 记录所有任务
    tasks = []
    
    for pheno_file in pheno_files:
        species, trait = parse_pheno_filename(pheno_file)
        
        if not trait:
            print(f"警告: 无法解析文件名 {pheno_file}，跳过")
            continue
        
        tasks.append((GENO_FILE, pheno_file, species, trait))
    
    print(f"准备运行 {len(tasks)} 个任务（依次执行）\n")
    
    # 询问用户是否继续
    print("任务列表:")
    for i, (geno_file, pheno_file, species, trait) in enumerate(tasks, 1):
        print(f"  {i}. {trait}")
    print()
    
    # 显示配置参数
    print("配置参数:")
    print(f"  - 预处理功能: 启用")
    print(f"  - 任务类型: regression")
    print(f"  - 表型标准化: {PREDICTION_PARAMS['standardize_phenotype']}")
    print(f"  - 使用默认参数: {PREDICTION_PARAMS['use_default_params']}")
    print(f"  - 重复次数: {PREDICTION_PARAMS['n_repeats']}")
    print(f"  - 交叉验证折数: {PREDICTION_PARAMS['n_splits']}")
    print(f"  - 并行任务数: {PREDICTION_PARAMS['parallel_jobs']}")
    print(f"  - 测试集比例: {PREDICTION_PARAMS['test_size']}")
    print(f"  - Stacking集成: {PREDICTION_PARAMS['use_stacking']}")
    print(f"  - 顶级模型数: {PREDICTION_PARAMS['top_n_models']}")
    print()
    
    response = input("是否开始运行所有任务? (y/n): ").strip().lower()
    if response != 'y':
        print("取消执行")
        return
    
    # 记录开始时间
    start_time = time.time()
    
    # 依次执行所有任务
    completed_tasks = []
    failed_tasks = []
    
    for i, (geno_file, pheno_file, species, trait) in enumerate(tasks, 1):
        print(f"\n{'#'*80}")
        print(f"# 正在执行任务 {i}/{len(tasks)}: {trait}")
        print(f"{'#'*80}")
        
        task_start = time.time()
        log_file, results_dir, return_code = run_task(geno_file, pheno_file, species, trait)
        task_end = time.time()
        task_duration = task_end - task_start
        
        task_info = {
            'species': species,
            'trait': trait,
            'log_file': log_file,
            'results_dir': results_dir,
            'duration': task_duration,
            'return_code': return_code
        }
        
        if return_code == 0:
            completed_tasks.append(task_info)
        else:
            failed_tasks.append(task_info)
        
        print(f"\n任务耗时: {task_duration/60:.2f} 分钟")
        print(f"剩余任务: {len(tasks) - i}")
        
        # 估算剩余时间
        if completed_tasks:
            avg_duration = sum(t['duration'] for t in completed_tasks) / len(completed_tasks)
            remaining_time = avg_duration * (len(tasks) - i)
            print(f"预计剩余时间: {remaining_time/60:.2f} 分钟 ({remaining_time/3600:.2f} 小时)")
    
    # 计算总耗时
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 输出总结
    print("\n" + "="*80)
    print("所有任务执行完毕！")
    print("="*80)
    
    print(f"\n总耗时: {total_duration/3600:.2f} 小时 ({total_duration/60:.2f} 分钟)")
    print(f"成功: {len(completed_tasks)} 个")
    print(f"失败: {len(failed_tasks)} 个")
    
    if completed_tasks:
        print("\n成功任务:")
        for i, task in enumerate(completed_tasks, 1):
            print(f"  {i}. {task['trait']} - {task['duration']/60:.2f} 分钟")
            print(f"     日志: {task['log_file']}")
            print(f"     结果: {task['results_dir']}")
            print(f"     预处理数据: {os.path.join(PREPROCESS_BASE_DIR, task['trait'])}")
    
    if failed_tasks:
        print("\n失败任务:")
        for i, task in enumerate(failed_tasks, 1):
            print(f"  {i}. {task['trait']} - 返回码: {task['return_code']}")
            print(f"     日志: {task['log_file']}")
    
    # 生成汇总报告
    summary_file = os.path.join(RESULTS_BASE_DIR, 'batch_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("批量基因组预测任务汇总报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"总耗时: {total_duration/3600:.2f} 小时\n")
        f.write(f"成功任务: {len(completed_tasks)}\n")
        f.write(f"失败任务: {len(failed_tasks)}\n\n")
        
        if completed_tasks:
            f.write("成功任务列表:\n")
            for task in completed_tasks:
                f.write(f"  - {task['trait']}: {task['duration']/60:.2f} 分钟\n")
        
        if failed_tasks:
            f.write("\n失败任务列表:\n")
            for task in failed_tasks:
                f.write(f"  - {task['trait']}: 返回码 {task['return_code']}\n")
    
    print(f"\n汇总报告已保存到: {summary_file}")

if __name__ == '__main__':
    main()
