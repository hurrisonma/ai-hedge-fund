#!/usr/bin/env python3
"""
业务成本损失函数测试版本
支持27种参数配置的串行测试
"""

import json
import os
import sys
import time
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
        
        # 输出训练历史详情（在最终汇总前）
        print("\n" + "=" * 80)
        print("📈 训练历史详情:")
        print("=" * 80)
        
        # 输出参数配置信息
        param_info = f"配置: 参数[误报={false_alarm}, 漏报={miss_change}, 奖励={correct_reward}]"
        print(param_info)
        
        # 由于train()方法没有返回详细的训练历史，我们输出简化的提示
        print("注意：详细的每轮训练信息已在训练过程中显示。")
        print("以下是最终测试集评估结果的详细信息：")
        
        # 输出最终测试集的详细评估信息（类似您要求的格式）
        for horizon_key, results in test_results.items():
            accuracy = results.get('accuracy', 0.0)
            precision = results.get('precision', 0.0)
            recall = results.get('recall', 0.0)
            f1 = results.get('f1', 0.0)
            cm = results.get('confusion_matrix')
            stable_accuracy = results.get('stable_accuracy', 0.0)
            change_accuracy = results.get('change_accuracy', 0.0)
            
            print(f"\n{horizon_key} 预测结果: 准确率={accuracy:.3f} | "
                  f"精确率={precision:.3f} | 召回率={recall:.3f} | F1={f1:.3f}")
            if cm is not None:
                cm_list = cm.tolist() if hasattr(cm, 'tolist') else cm
                print(f"  混淆矩阵: {cm_list}")
            print(f"  稳定类准确率: {stable_accuracy:.3f} | "
                  f"变化类准确率: {change_accuracy:.3f}")
        
        print("=" * 80)
        
        # 修复：从5分钟预测结果中提取关键指标
        result_5min = test_results.get('5min', {})
        change_accuracy = result_5min.get('change_accuracy', 0.0)
        stable_accuracy = result_5min.get('stable_accuracy', 0.0)
        
        # 计算加权评分
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
            'accuracy': result_5min.get('accuracy', 0.0),
            'precision': result_5min.get('precision', 0.0),
            'recall': result_5min.get('recall', 0.0),
            'f1_score': result_5min.get('f1', 0.0),  # 注意这里是'f1'不是'f1_score'
            
            # 类别准确率
            'stable_accuracy': stable_accuracy,
            'change_accuracy': change_accuracy,
            'weighted_score': weighted_score,
            
            # 错误分析
            'false_positives': 0,  # 需要从混淆矩阵计算
            'false_negatives': 0,  # 需要从混淆矩阵计算
            'catastrophic_error_rate': 0.0,  # 需要从混淆矩阵计算
            
            # 混淆矩阵
            'confusion_matrix': [[0, 0], [0, 0]],  # 将在下面更新
            
            # 训练信息
            'training_time': training_time,
            'final_epoch': len(training_history) if training_history else 0,
            'status': 'success'
        }
        
        # 处理混淆矩阵数据
        cm_data = result_5min.get('confusion_matrix', [[0, 0], [0, 0]])
        if hasattr(cm_data, 'tolist'):
            cm_data = cm_data.tolist()
        result['confusion_matrix'] = cm_data
        
        # 从混淆矩阵计算详细错误分析
        cm = result_5min.get('confusion_matrix')
        if cm is not None:
            if hasattr(cm, 'tolist'):
                cm = cm.tolist()
            
            # 计算false_positives和false_negatives
            if len(cm) >= 2 and len(cm[0]) >= 2:
                result['false_positives'] = int(cm[0][1])  # 稳定->变化
            if len(cm) >= 2 and len(cm[1]) >= 1:
                result['false_negatives'] = int(cm[1][0])  # 变化->稳定
            
            # 计算灾难性错误率
            total_samples = sum(sum(row) for row in cm)
            catastrophic_errors = (result['false_positives'] +
                                   result['false_negatives'])
            if total_samples > 0:
                result['catastrophic_error_rate'] = float(
                    catastrophic_errors / total_samples
                )
        
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


class BusinessCostTest:
    """业务成本参数测试类（串行版本）"""
    
    def __init__(self):
        self.false_alarm_costs = [1, 2, 3]
        self.miss_change_costs = [6, 7, 8] 
        self.correct_rewards = [0.2, 0.3, 0.4]
        
        # 生成所有组合
        self.param_combinations = list(product(
            self.false_alarm_costs,
            self.miss_change_costs, 
            self.correct_rewards
        ))
        
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🧪 准备串行测试 {len(self.param_combinations)} 种参数组合")
        
    def run_all_tests(self):
        """运行所有参数组合测试（串行版本）"""
        print("🚀 开始BusinessCost参数测试")
        print("=" * 80)
        
        start_time = time.time()
        
        # 串行执行每个参数组合
        for i, params in enumerate(self.param_combinations, 1):
            num_combinations = len(self.param_combinations)
            progress_percent = (i / num_combinations) * 100
            print(f"\n📊 进度: {i}/{num_combinations} "
                  f"({progress_percent:.1f}%)")
            print(f"🔬 正在测试参数组合: {params}")
            
            try:
                result = run_single_test_worker(params)
                self.results.append(result)
                
                # 实时保存结果
                self.save_intermediate_results()
                
            except Exception as e:
                print(f"❌ 参数组合 {params} 执行异常: {str(e)}")
                # 添加失败记录
                failed_result = {
                    'false_alarm_cost': params[0],
                    'miss_change_cost': params[1],
                    'correct_reward': params[2],
                    'param_signature': (
                        f"f{params[0]}_m{params[1]}_r{params[2]}"
                    ),
                    'status': 'failed',
                    'error': str(e),
                    'change_accuracy': 0.0,
                    'stable_accuracy': 0.0,
                    'weighted_score': 0.0
                }
                self.results.append(failed_result)
        
        total_time = time.time() - start_time
        print(f"\n🎉 所有测试完成！总耗时: {total_time/60:.1f}分钟")

    def save_intermediate_results(self):
        """保存中间结果"""
        if not self.results:
            return
            
        json_file = (
            f"outputs/business_cost_test_{self.timestamp}.json"
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
        
        # 首先输出所有组合的详细信息
        print("\n" + "=" * 80)
        print("📈 所有参数组合详细结果:")
        print("=" * 80)
        
        for i, (_, row) in enumerate(success_df.iterrows(), 1):
            false_alarm = row['false_alarm_cost']
            miss_change = row['miss_change_cost']
            correct_reward = row['correct_reward']
            
            print(f"\n{i:2d}. 配置: 参数[误报={false_alarm}, 漏报={miss_change}, 奖励={correct_reward}]")
            
            # 从结果中提取评估信息（注意：这些是测试集的结果，不是每轮训练的）
            accuracy = row.get('accuracy', 0.0)
            precision = row.get('precision', 0.0)
            recall = row.get('recall', 0.0)
            f1_score = row.get('f1_score', 0.0)
            stable_accuracy = row.get('stable_accuracy', 0.0)
            change_accuracy = row.get('change_accuracy', 0.0)
            
            print(f"    最终测试结果: 准确率={accuracy:.3f} | "
                  f"精确率={precision:.3f} | 召回率={recall:.3f} | F1={f1_score:.3f}")
            
            # 混淆矩阵
            cm = row.get('confusion_matrix', [[0, 0], [0, 0]])
            if isinstance(cm, str):
                try:
                    import ast
                    cm = ast.literal_eval(cm)
                except:
                    cm = [[0, 0], [0, 0]]
            print(f"    混淆矩阵: {cm}")
            
            print(f"    稳定类准确率: {stable_accuracy:.3f} | "
                  f"变化类准确率: {change_accuracy:.3f}")
            
            # 加权评分和其他指标
            weighted_score = row.get('weighted_score', 0.0)
            training_time = row.get('training_time', 0.0)
            print(f"    加权评分: {weighted_score:.3f} | "
                  f"训练时间: {training_time/60:.1f}分钟")
        
        print("=" * 80)
            
        print("\n📊 BusinessCost测试结果分析")
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
            f"outputs/business_cost_report_{self.timestamp}.csv"
        )
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
    print("  - 执行方式: 串行")
    print("=" * 60)
    
    # 创建测试实例
    tester = BusinessCostTest()
    
    # 运行所有测试
    tester.run_all_tests()
    
    # 生成分析报告
    tester.generate_report()
    
    print("\n🎉 BusinessCost参数优化测试完成！")


if __name__ == "__main__":
    main() 