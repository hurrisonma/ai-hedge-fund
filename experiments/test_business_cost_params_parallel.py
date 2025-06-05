#!/usr/bin/env python3
"""
业务成本损失函数并行测试版本
支持多进程运行以加速27种参数配置的测试

使用GPU时建议2-3个进程，避免显存不足
使用CPU时可以使用更多进程
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product

import pandas as pd

# 将项目根目录添加到sys.path，以便导入自定义模块
# 建议在所有标准库和第三方库导入之后，自定义模块导入之前进行
sys.path.append('.')

from main import DeepLearningExperiment
from training.config import ExperimentConfig


def run_single_test_worker(params):
    """单个参数组合的工作进程"""
    false_alarm, miss_change, correct_reward = params
    
    # 打印进程ID以验证多进程运行
    print(f"💼 工作进程 PID: {os.getpid()} 正在处理参数: {params}")
    
    print(f"\n🔬 进程启动: 误报={false_alarm}, 漏报={miss_change}, 奖励={correct_reward}")
    
    # 创建配置
    config = ExperimentConfig()
    config.loss_function_type = "business_cost"
    
    # 设置business_cost参数
    config.loss_function_params["business_cost"] = {
        "false_alarm_cost": false_alarm,
        "miss_change_cost": miss_change,
        "correct_reward": correct_reward
    }
    
    # 输出目录配置
    param_name = (
        f"business_cost_f{false_alarm}_m{miss_change}_r{correct_reward}"
    )
    config.output_dir = f"outputs/{param_name}"
    config.model_save_dir = f"outputs/{param_name}/models"
    config.log_dir = f"outputs/{param_name}/logs"
    config.plot_dir = f"outputs/{param_name}/plots"
    
    # 手动创建所有必要的目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)
    
    # 简化训练配置
    config.max_epochs = 10
    config.early_stopping_patience = 999
    config.multi_metric_early_stopping['enabled'] = False
    
    try:
        # 运行实验
        start_time = time.time()
        experiment = DeepLearningExperiment(config)
        experiment.prepare_data()
        experiment.build_model()
        
        # 训练
        training_history = experiment.train()
        
        # 评估
        test_results = experiment.evaluate()
        
        training_time = time.time() - start_time
        
        # 计算加权评分
        change_accuracy = test_results.get('change_accuracy', 0.0)
        stable_accuracy = test_results.get('stable_accuracy', 0.0)
        weighted_score = change_accuracy * 2.0 + stable_accuracy * 1.0
        
        # 提取关键指标
        result = {
            'false_alarm_cost': false_alarm,
            'miss_change_cost': miss_change,
            'correct_reward': correct_reward,
            'param_signature': (
                f"f{false_alarm}_m{miss_change}_r{correct_reward}"
            ),
            
            # 核心性能指标
            'accuracy': test_results.get('accuracy', 0.0),
            'precision': test_results.get('precision', 0.0),
            'recall': test_results.get('recall', 0.0),
            'f1_score': test_results.get('f1_score', 0.0),
            
            # 类别准确率
            'stable_accuracy': stable_accuracy,
            'change_accuracy': change_accuracy,
            'weighted_score': weighted_score,
            
            # 错误分析
            'false_positives': test_results.get('false_positives', 0),
            'false_negatives': test_results.get('false_negatives', 0),
            'catastrophic_error_rate': test_results.get(
                'catastrophic_error_rate', 0.0
            ),
            
            # 混淆矩阵
            'confusion_matrix': test_results.get(
                'confusion_matrix', [[0, 0], [0, 0]]
            ),
            
            # 训练信息
            'training_time': training_time,
            'final_epoch': len(training_history) if training_history else 0,
            'status': 'success'
        }
        
        print(f"✅ 完成 {param_name}: 变化类={change_accuracy:.1%}, "
              f"稳定类={stable_accuracy:.1%}, 加权评分={weighted_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ 失败 {param_name}: {str(e)}")
        return {
            'false_alarm_cost': false_alarm,
            'miss_change_cost': miss_change,
            'correct_reward': correct_reward,
            'param_signature': (
                f"f{false_alarm}_m{miss_change}_r{correct_reward}"
            ),
            'status': 'failed',
            'error': str(e),
            'change_accuracy': 0.0,
            'stable_accuracy': 0.0,
            'weighted_score': 0.0
        }


class ParallelBusinessCostTest:
    """并行业务成本参数测试类"""
    
    def __init__(self, max_workers=2):
        self.false_alarm_costs = [1, 2, 3]
        self.miss_change_costs = [6, 7, 8] 
        self.correct_rewards = [0.2, 0.3, 0.4]
        
        # 生成所有组合
        self.param_combinations = list(product(
            self.false_alarm_costs,
            self.miss_change_costs, 
            self.correct_rewards
        ))
        
        self.max_workers = max_workers
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🧪 准备并行测试 {len(self.param_combinations)} 种参数组合")
        print(f"🔄 使用 {max_workers} 个进程")
        
    def run_all_tests(self):
        """运行所有参数组合测试（并行版本）"""
        print("🚀 开始并行BusinessCost参数测试")
        print("=" * 80)
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_params = {
                executor.submit(run_single_test_worker, params): params 
                for params in self.param_combinations
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    completed += 1
                    
                    num_combinations = len(self.param_combinations)
                    progress_percent = (completed / num_combinations) * 100
                    print(f"📊 进度: {completed}/{num_combinations} "
                          f"({progress_percent:.1f}%)")
                    
                    # 实时保存结果
                    self.save_intermediate_results()
                    
                except Exception as e:
                    print(f"❌ 任务 {params} 执行异常: {str(e)}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 所有测试完成！总耗时: {total_time/60:.1f}分钟")
        
    def save_intermediate_results(self):
        """保存中间结果"""
        if not self.results:
            return
            
        json_file = (
            f"outputs/business_cost_test_parallel_{self.timestamp}.json"
        )
        os.makedirs("outputs", exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
    def generate_report(self):
        """生成对比报告"""
        if not self.results:
            print("❌ 没有测试结果可分析")
            return
            
        # 转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 只分析成功的结果
        success_df = df[df['status'] == 'success'].copy()
        
        if success_df.empty:
            print("❌ 没有成功的测试结果")
            return
            
        print("\n📊 BusinessCost并行测试结果分析")
        print("=" * 80)
        
        # 按加权评分排序
        print("\n🏆 按加权评分排序 (变化类权重×2 + 稳定类权重×1):")
        top_weighted = success_df.nlargest(10, 'weighted_score')
        for i, (_, row) in enumerate(top_weighted.iterrows(), 1):
            print(f"  {i:2d}. {row['param_signature']}: "
                  f"得分={row['weighted_score']:.3f} | "
                  f"变化类={row['change_accuracy']:.1%}, "
                  f"稳定类={row['stable_accuracy']:.1%}")
        
        # 按变化类准确率排序
        print("\n🎯 按变化类准确率排序 (前10名):")
        top_change = success_df.nlargest(10, 'change_accuracy')
        for _, row in top_change.iterrows():
            print(f"  {row['param_signature']}: "
                  f"变化类={row['change_accuracy']:.1%}, "
                  f"稳定类={row['stable_accuracy']:.1%}, "
                  f"加权评分={row['weighted_score']:.3f}")
        
        # 最优配置推荐
        print("\n🏆 最优配置推荐:")
        best_weighted = success_df.loc[success_df['weighted_score'].idxmax()]
        print(f"  🥇 最佳加权评分: {best_weighted['param_signature']}")
        print(f"    - 加权评分: {best_weighted['weighted_score']:.3f}")
        print(f"    - 变化类准确率: {best_weighted['change_accuracy']:.1%}")
        print(f"    - 稳定类准确率: {best_weighted['stable_accuracy']:.1%}")
        
        best_change = success_df.loc[success_df['change_accuracy'].idxmax()]
        print(f"  🥈 最佳变化类准确率: {best_change['param_signature']} "
              f"({best_change['change_accuracy']:.1%})")
        
        # 保存CSV报告
        csv_file = (
            f"outputs/business_cost_parallel_report_{self.timestamp}.csv"
        )
        success_df.to_csv(csv_file, index=False)
        print(f"\n💾 详细结果已保存到: {csv_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BusinessCost并行参数优化测试')
    parser.add_argument('--workers', type=int, default=2, 
                        help='并行进程数 (GPU建议2-3个，CPU可更多)')
    args = parser.parse_args()
    
    print("🚀 BusinessCost损失函数并行参数优化测试")
    print("测试参数:")
    print("  - 误报成本: [1, 2, 3]")
    print("  - 漏报成本: [6, 7, 8]") 
    print("  - 正确奖励: [0.2, 0.3, 0.4]")
    print("  - 总组合数: 27种")
    print(f"  - 并行进程数: {args.workers}")
    print("=" * 60)
    
    # 创建测试实例
    tester = ParallelBusinessCostTest(max_workers=args.workers)
    
    # 运行所有测试
    tester.run_all_tests()
    
    # 生成分析报告
    tester.generate_report()
    
    print("\n🎉 BusinessCost并行参数优化测试完成！")


if __name__ == "__main__":
    main() 