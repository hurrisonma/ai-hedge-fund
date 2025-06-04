#!/usr/bin/env python3
"""
业务成本损失函数排列组合测试
测试27种参数配置的性能对比

参数范围:
- 误报成本: [1, 2, 3]
- 漏报成本: [6, 7, 8] 
- 正确奖励: [0.2, 0.3, 0.4]
"""

import json
import os
import sys
import time
from datetime import datetime
from itertools import product
from typing import Dict

import pandas as pd

sys.path.append('.')

from main import DeepLearningExperiment
from training.config import ExperimentConfig


class BusinessCostParameterTest:
    """业务成本参数测试类"""
    
    def __init__(self):
        self.base_config = ExperimentConfig()
        self.base_config.loss_function_type = "business_cost"
        
        # 测试参数范围
        self.false_alarm_costs = [1, 2, 3]
        self.miss_change_costs = [6, 7, 8] 
        self.correct_rewards = [0.2, 0.3, 0.4]
        
        # 生成所有组合
        self.param_combinations = list(product(
            self.false_alarm_costs,
            self.miss_change_costs, 
            self.correct_rewards
        ))
        
        print(f"🧪 准备测试 {len(self.param_combinations)} 种参数组合")
        
        # 结果存储
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def calculate_weighted_score(self, change_accuracy: float, 
                                 stable_accuracy: float) -> float:
        """
        统一加权评分函数：变化类权重2倍，稳定类权重1倍
        
        Args:
            change_accuracy: 变化类准确率 (0-1)
            stable_accuracy: 稳定类准确率 (0-1)
            
        Returns:
            加权得分 (0-3)
        """
        return change_accuracy * 2.0 + stable_accuracy * 1.0
        
    def create_config(self, false_alarm: float, miss_change: float, 
                      correct_reward: float) -> ExperimentConfig:
        """创建特定参数的配置"""
        config = ExperimentConfig()
        config.loss_function_type = "business_cost"
        
        # 设置business_cost参数
        config.loss_function_params["business_cost"] = {
            "false_alarm_cost": false_alarm,
            "miss_change_cost": miss_change,
            "correct_reward": correct_reward
        }
        
        # 输出目录配置
        param_name = f"business_cost_f{false_alarm}_m{miss_change}_r{correct_reward}"
        config.output_dir = f"outputs/{param_name}"
        config.model_save_dir = f"outputs/{param_name}/models"
        config.log_dir = f"outputs/{param_name}/logs"
        config.plot_dir = f"outputs/{param_name}/plots"
        
        # 手动创建所有必要的目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.plot_dir, exist_ok=True)
        
        # 简化训练配置（加速测试）
        config.max_epochs = 10  # 改为10轮训练
        config.early_stopping_patience = 999  # 禁用早停
        config.multi_metric_early_stopping['enabled'] = False  # 禁用多指标早停
        
        return config
        
    def run_single_test(self, false_alarm: float, miss_change: float, 
                        correct_reward: float) -> Dict:
        """运行单个参数组合测试"""
        print(f"\n🔬 测试配置: 误报={false_alarm}, 漏报={miss_change}, "
              f"奖励={correct_reward}")
        
        # 创建配置
        config = self.create_config(false_alarm, miss_change, correct_reward)
        
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
            
            # 提取关键指标
            result = {
                'false_alarm_cost': false_alarm,
                'miss_change_cost': miss_change,
                'correct_reward': correct_reward,
                'param_signature': f"f{false_alarm}_m{miss_change}_r{correct_reward}",
                
                # 核心性能指标
                'accuracy': test_results.get('accuracy', 0.0),
                'precision': test_results.get('precision', 0.0),
                'recall': test_results.get('recall', 0.0),
                'f1_score': test_results.get('f1_score', 0.0),
                
                # 类别准确率（最重要）
                'stable_accuracy': test_results.get('stable_accuracy', 0.0),
                'change_accuracy': test_results.get('change_accuracy', 0.0),
                
                # 统一加权评分（2:1权重）
                'weighted_score': self.calculate_weighted_score(
                    test_results.get('change_accuracy', 0.0),
                    test_results.get('stable_accuracy', 0.0)
                ),
                
                # 错误分析
                'false_positives': test_results.get('false_positives', 0),
                'false_negatives': test_results.get('false_negatives', 0),
                'catastrophic_error_rate': test_results.get('catastrophic_error_rate', 0.0),
                
                # 混淆矩阵
                'confusion_matrix': test_results.get('confusion_matrix', [[0, 0], [0, 0]]),
                
                # 训练信息
                'training_time': training_time,
                'final_epoch': len(training_history) if training_history else 0,
                'status': 'success'
            }
            
            print(f"✅ 测试完成: 变化类准确率={result['change_accuracy']:.1%}, "
                  f"稳定类准确率={result['stable_accuracy']:.1%}, "
                  f"加权评分={result['weighted_score']:.3f}")
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            result = {
                'false_alarm_cost': false_alarm,
                'miss_change_cost': miss_change,
                'correct_reward': correct_reward,
                'param_signature': f"f{false_alarm}_m{miss_change}_r{correct_reward}",
                'status': 'failed',
                'error': str(e),
                'change_accuracy': 0.0,
                'stable_accuracy': 0.0
            }
        
        return result
        
    def run_all_tests(self):
        """运行所有参数组合测试"""
        print(f"🚀 开始Business Cost参数对比测试 - {len(self.param_combinations)}种组合")
        print("=" * 80)
        
        for i, (false_alarm, miss_change, correct_reward) in enumerate(self.param_combinations):
            print(f"\n📊 进度: {i+1}/{len(self.param_combinations)}")
            
            result = self.run_single_test(false_alarm, miss_change, correct_reward)
            self.results.append(result)
            
            # 实时保存结果（防止中途失败丢失数据）
            self.save_intermediate_results()
            
        print("\n🎉 所有测试完成！")
        
    def save_intermediate_results(self):
        """保存中间结果"""
        if not self.results:
            return
            
        # JSON格式详细结果
        json_file = f"outputs/business_cost_test_{self.timestamp}.json"
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
            
        print("\n📊 BusinessCost参数测试结果分析")
        print("=" * 80)
        
        # 🏆 按加权评分排序（主要标准）
        print("\n🏆 按加权评分排序 (变化类权重×2 + 稳定类权重×1):")
        top_weighted = success_df.nlargest(10, 'weighted_score')
        for i, (_, row) in enumerate(top_weighted.iterrows(), 1):
            print(f"  {i:2d}. {row['param_signature']}: "
                  f"得分={row['weighted_score']:.3f} | "
                  f"变化类={row['change_accuracy']:.1%}, "
                  f"稳定类={row['stable_accuracy']:.1%}")
        
        # 1. 按变化类准确率排序
        print("\n🎯 按变化类准确率排序 (前10名):")
        top_change = success_df.nlargest(10, 'change_accuracy')
        for _, row in top_change.iterrows():
            print(f"  {row['param_signature']}: 变化类={row['change_accuracy']:.1%}, "
                  f"稳定类={row['stable_accuracy']:.1%}, "
                  f"加权评分={row['weighted_score']:.3f}")
        
        # 2. 参数影响分析
        print("\n📈 参数影响分析:")
        
        # 误报成本影响
        print("  误报成本影响:")
        for cost in self.false_alarm_costs:
            subset = success_df[success_df['false_alarm_cost'] == cost]
            if not subset.empty:
                avg_change = subset['change_accuracy'].mean()
                avg_weighted = subset['weighted_score'].mean()
                print(f"    误报成本={cost}: 平均变化类准确率={avg_change:.1%}, "
                      f"平均加权评分={avg_weighted:.3f}")
        
        # 漏报成本影响
        print("  漏报成本影响:")
        for cost in self.miss_change_costs:
            subset = success_df[success_df['miss_change_cost'] == cost]
            if not subset.empty:
                avg_change = subset['change_accuracy'].mean()
                avg_weighted = subset['weighted_score'].mean()
                print(f"    漏报成本={cost}: 平均变化类准确率={avg_change:.1%}, "
                      f"平均加权评分={avg_weighted:.3f}")
        
        # 奖励影响
        print("  正确奖励影响:")
        for reward in self.correct_rewards:
            subset = success_df[success_df['correct_reward'] == reward]
            if not subset.empty:
                avg_change = subset['change_accuracy'].mean()
                avg_weighted = subset['weighted_score'].mean()
                print(f"    正确奖励={reward}: 平均变化类准确率={avg_change:.1%}, "
                      f"平均加权评分={avg_weighted:.3f}")
        
        # 4. 最优配置推荐
        print("\n🏆 最优配置推荐:")
        
        # 按加权评分选择最优（主要推荐）
        best_weighted = success_df.loc[success_df['weighted_score'].idxmax()]
        print(f"  🥇 最佳加权评分: {best_weighted['param_signature']}")
        print(f"    - 加权评分: {best_weighted['weighted_score']:.3f}")
        print(f"    - 变化类准确率: {best_weighted['change_accuracy']:.1%}")
        print(f"    - 稳定类准确率: {best_weighted['stable_accuracy']:.1%}")
        
        # 其他参考指标
        best_change = success_df.loc[success_df['change_accuracy'].idxmax()]
        
        print(f"  🥈 最佳变化类准确率: {best_change['param_signature']} "
              f"({best_change['change_accuracy']:.1%})")
        
        # 保存CSV报告
        csv_file = f"outputs/business_cost_report_{self.timestamp}.csv"
        success_df.to_csv(csv_file, index=False)
        print(f"\n💾 详细结果已保存到: {csv_file}")

def main():
    """主函数"""
    print("🚀 BusinessCost损失函数参数优化测试")
    print("测试参数:")
    print("  - 误报成本: [1, 2, 3]")
    print("  - 漏报成本: [6, 7, 8]") 
    print("  - 正确奖励: [0.2, 0.3, 0.4]")
    print("  - 总组合数: 27种")
    print("=" * 60)
    
    # 创建测试实例
    tester = BusinessCostParameterTest()
    
    # 运行所有测试
    tester.run_all_tests()
    
    # 生成分析报告
    tester.generate_report()
    
    print("\n🎉 BusinessCost参数优化测试完成！")

if __name__ == "__main__":
    main() 