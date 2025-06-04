#!/usr/bin/env python3
"""
🧪 损失函数自动测试脚本
快速测试不同损失函数的效果并生成对比报告
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import DeepLearningExperiment
from training.config import ExperimentConfig


class LossFunctionTester:
    """损失函数测试器"""
    
    def __init__(self, test_epochs: int = 3):
        self.test_epochs = test_epochs
        self.results = []
        self.start_time = datetime.now()
        
    def test_loss_function(self, loss_type: str, 
                          weight_config: float = None,
                          custom_params: Dict = None) -> Dict:
        """测试单个损失函数"""
        weight_suffix = f"_w{weight_config}" if weight_config else ""
        test_name = f"{loss_type}{weight_suffix}"
        
        print(f"\n{'='*60}")
        print(f"🧪 测试: {test_name}")
        if weight_config:
            print(f"📊 变化类权重: {weight_config}")
        print(f"{'='*60}")
        
        # 创建配置
        config = ExperimentConfig()
        config.loss_function_type = loss_type
        config.max_epochs = self.test_epochs
        
        # 🎯 动态设置class_weights
        if weight_config and loss_type == 'binary_cross_entropy':
            config.class_weights = [1.0, weight_config]
            print(f"📈 使用权重配置: [1.0, {weight_config}]")
        
        # 🎯 为每个配置创建独立的输出目录
        base_output_dir = config.output_dir
        config_output_dir = f"{base_output_dir}/{test_name}"
        
        # 更新所有输出路径
        config.output_dir = config_output_dir
        config.model_save_dir = f"{config_output_dir}/models"
        config.log_dir = f"{config_output_dir}/logs"
        config.plot_dir = f"{config_output_dir}/plots"
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.plot_dir, exist_ok=True)
        
        # 自定义参数
        if custom_params:
            for key, value in custom_params.items():
                if key == 'loss_function_params':
                    # 更新损失函数参数
                    for loss_func_name, params in value.items():
                        if loss_func_name in config.loss_function_params:
                            config.loss_function_params[loss_func_name].update(params)
                elif hasattr(config, key):
                    setattr(config, key, value)
        
        # 运行实验
        start_time = time.time()
        try:
            experiment = DeepLearningExperiment(config)
            results = experiment.run_experiment()
            
            # 提取关键指标
            key_metrics = self._extract_key_metrics(results, loss_type)
            key_metrics['training_time'] = time.time() - start_time
            key_metrics['status'] = 'success'
            key_metrics['model_save_dir'] = config.model_save_dir  # 记录模型保存路径
            
            return key_metrics
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return {
                'loss_function': loss_type,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time,
                'model_save_dir': config.model_save_dir if 'config' in locals() else None
            }
    
    def _extract_key_metrics(self, results: Dict, 
                           loss_type: str) -> Dict:
        """提取关键评估指标"""
        # 假设results包含各时间尺度的结果
        metrics = {'loss_function': loss_type}
        
        # 提取5分钟预测结果（主要关注）
        if '5min' in results:
            result_5min = results['5min']
            
            # 基础指标
            metrics.update({
                'accuracy': float(result_5min.get('accuracy', 0)),
                'precision': float(result_5min.get('precision', 0)),
                'recall': float(result_5min.get('recall', 0)),
                'f1_score': float(result_5min.get('f1', 0)),
            })
            
            # 混淆矩阵
            cm = result_5min.get('confusion_matrix')
            if cm is not None:
                # 计算各类别准确率
                stable_correct = int(cm[0, 0])
                stable_total = int(cm[0, 0] + cm[0, 1])
                change_correct = int(cm[1, 1])
                change_total = int(cm[1, 0] + cm[1, 1])
                
                stable_accuracy = (stable_correct / stable_total 
                                 if stable_total > 0 else 0.0)
                change_accuracy = (change_correct / change_total 
                                 if change_total > 0 else 0.0)
                
                # 🎯 新增：计算灾难性错误率
                total_samples = stable_total + change_total
                catastrophic_errors = cm[0, 1] + cm[1, 0]  # 互相误判
                catastrophic_rate = (catastrophic_errors / total_samples 
                                   if total_samples > 0 else 0.0)
                
                metrics.update({
                    'stable_accuracy': float(stable_accuracy),
                    'change_accuracy': float(change_accuracy),
                    'false_positives': int(cm[0, 1]),  # 稳定误判为变化
                    'false_negatives': int(cm[1, 0]),  # 变化误判为稳定
                    'catastrophic_error_rate': float(catastrophic_rate),
                    'confusion_matrix': [[int(cm[0, 0]), int(cm[0, 1])],
                                       [int(cm[1, 0]), int(cm[1, 1])]]
                })
                
                # 🎯 新增：提取更多关键指标
                if 'balanced_class_accuracy' in result_5min:
                    metrics['balanced_class_accuracy'] = float(result_5min['balanced_class_accuracy'])
                if 'composite_score' in result_5min:
                    metrics['composite_score'] = float(result_5min['composite_score'])
                if 'is_failed_model' in result_5min:
                    metrics['is_failed_model'] = bool(result_5min['is_failed_model'])
        
        return metrics
    
    def run_all_tests(self) -> pd.DataFrame:
        """运行所有损失函数测试"""
        print(f"🚀 开始损失函数对比测试")
        print(f"测试轮数: {self.test_epochs}")
        print(f"开始时间: {self.start_time}")
        
        # 定义测试方案 - 只测试binary_cross_entropy的不同权重配置
        test_cases = [
            {
                'loss_type': 'binary_cross_entropy',
                'description': '标准交叉熵-权重6.5',
                'weight_config': 6.5,
                'params': {}
            },
            {
                'loss_type': 'binary_cross_entropy', 
                'description': '标准交叉熵-权重7.0',
                'weight_config': 7.0,
                'params': {}
            },
            {
                'loss_type': 'binary_cross_entropy',
                'description': '标准交叉熵-权重7.5', 
                'weight_config': 7.5,
                'params': {}
            },
            {
                'loss_type': 'binary_cross_entropy',
                'description': '标准交叉熵-权重8.0',
                'weight_config': 8.0,
                'params': {}
            },
            # {
            #     'loss_type': 'business_cost',
            #     'description': '业务成本驱动（调整漏报成本）',
            #     'params': {
            #         'loss_function_params': {
            #             "business_cost": {
            #                 "false_alarm_cost": 1.0,    # 误报成本(稳定->变化)
            #                 "miss_change_cost": 8.0,    # 漏报成本(变化->稳定)
            #                 "correct_reward": 0.2,      # 正确预测奖励
            #             }
            #         }
            #     }
            # },
        ]
        
        # 执行测试
        results = []
        total_tests = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎯 进度: {i}/{total_tests}")
            print(f"📝 测试配置: {test_case['description']}")
            
            try:
                # 传入weight_config参数
                weight_config = test_case.get('weight_config', None)
                result = self.test_loss_function(
                    test_case['loss_type'], 
                    weight_config,
                    test_case['params']
                )
                
                # 添加描述信息到结果中
                result['description'] = test_case['description']
                result['weight_config'] = weight_config
                results.append(result)
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                continue
        
        # 生成报告
        return self._generate_report(results)
    
    def _generate_report(self, results: List[Dict]) -> pd.DataFrame:
        """生成测试报告"""
        print(f"\n{'='*60}")
        print(f"📊 测试报告生成")
        print(f"{'='*60}")
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存详细结果
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        detail_file = f"outputs/loss_function_test_{timestamp}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成简化报告
        if len(df) > 0:
            report_file = f"outputs/loss_function_report_{timestamp}.csv"
            
            # 选择关键列
            key_columns = [
                'loss_function', 'status', 'accuracy', 'precision', 
                'recall', 'f1_score', 'stable_accuracy', 'change_accuracy',
                'false_positives', 'false_negatives', 'training_time'
            ]
            
            available_columns = [col for col in key_columns if col in df.columns]
            summary_df = df[available_columns].copy()
            
            # 保存报告
            summary_df.to_csv(report_file, index=False)
            print(f"📄 详细结果: {detail_file}")
            print(f"📋 简化报告: {report_file}")
            
            # 打印摘要
            self._print_summary(summary_df)
            
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """打印测试摘要"""
        print(f"\n📈 测试摘要:")
        print("-" * 50)
        
        successful_tests = df[df['status'] == 'success']
        
        if len(successful_tests) > 0:
            # 🎯 新增：检查失败模型
            failed_models = []
            valid_models = []
            
            for _, row in successful_tests.iterrows():
                if row.get('is_failed_model', False):
                    failed_models.append(row['loss_function'])
                else:
                    valid_models.append(row)
            
            if failed_models:
                print("❌ 失败模型（不满足基本要求）:")
                for model_name in failed_models:
                    print(f"  {model_name}")
                print()
            
            # 🎯 按综合评分排序（如果有的话）
            if 'composite_score' in successful_tests.columns:
                # 只对有效模型排序
                valid_df = pd.DataFrame(valid_models) if valid_models else pd.DataFrame()
                if len(valid_df) > 0:
                    top_performers = valid_df.nlargest(3, 'composite_score')
                    
                    print("🏆 综合评分排名（有效模型）:")
                    for i, (_, row) in enumerate(top_performers.iterrows(), 1):
                        print(f"  {i}. {row['loss_function']}: 综合评分={row['composite_score']:.3f}")
                    print()
            
            # 按F1分数排序（备选排序）
            elif 'f1_score' in successful_tests.columns:
                valid_df = pd.DataFrame(valid_models) if valid_models else pd.DataFrame()
                if len(valid_df) > 0:
                    top_performers = valid_df.nlargest(3, 'f1_score')
                    
                    print("🏆 F1分数排名（有效模型）:")
                    for i, (_, row) in enumerate(top_performers.iterrows(), 1):
                        print(f"  {i}. {row['loss_function']}: F1={row['f1_score']:.3f}")
                    print()
            
            # 🎯 详细类别准确率对比
            if 'stable_accuracy' in successful_tests.columns:
                print(f"📊 详细评估对比:")
                for _, row in successful_tests.iterrows():
                    model_name = row['loss_function']
                    stable_acc = row.get('stable_accuracy', 0)
                    change_acc = row.get('change_accuracy', 0)
                    catastrophic_rate = row.get('catastrophic_error_rate', 0)
                    model_path = row.get('model_save_dir', 'N/A')
                    
                    # 状态标识
                    if row.get('is_failed_model', False):
                        status = "❌ 失败"
                    elif change_acc > 0.5 and stable_acc > 0.6:
                        status = "✅ 优秀"
                    elif change_acc > 0.3:
                        status = "⚠️  一般"
                    else:
                        status = "🔴 较差"
                    
                    print(f"  {model_name}: {status}")
                    print(f"    稳定类: {stable_acc:.3f}, 变化类: {change_acc:.3f}")
                    print(f"    灾难错误率: {catastrophic_rate:.3f}")
                    print(f"    模型路径: {model_path}")
                    
                    # 显示综合评分或平衡准确率
                    if 'composite_score' in row:
                        print(f"    综合评分: {row['composite_score']:.3f}")
                    elif 'balanced_class_accuracy' in row:
                        print(f"    平衡准确率: {row['balanced_class_accuracy']:.3f}")
                    print()
        
        failed_tests = df[df['status'] == 'failed']
        if len(failed_tests) > 0:
            print(f"❌ 运行失败的测试:")
            for _, row in failed_tests.iterrows():
                print(f"  {row['loss_function']}: {row.get('error', 'Unknown error')}")

        # 🎯 新增：推荐最佳模型
        if len(successful_tests) > 0:
            print(f"\n🎯 推荐结论:")
            
            # 找出最佳有效模型
            valid_models_df = successful_tests[
                successful_tests.get('is_failed_model', pd.Series([True]*len(successful_tests))) == False
            ]
            
            if len(valid_models_df) > 0:
                if 'composite_score' in valid_models_df.columns:
                    best_model = valid_models_df.loc[valid_models_df['composite_score'].idxmax()]
                    print(f"  📊 推荐模型: {best_model['loss_function']}")
                    print(f"  📈 综合评分: {best_model['composite_score']:.3f}")
                    print(f"  🎯 变化类准确率: {best_model.get('change_accuracy', 0):.3f}")
                else:
                    print("  ⚠️  所有模型都存在问题，建议进一步调优")
            else:
                print("  ❌ 没有通过基本要求的模型，需要重新设计")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='损失函数对比测试')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='测试轮数 (默认: 3)')
    parser.add_argument('--single', type=str, 
                       help='只测试单个损失函数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    tester = LossFunctionTester(test_epochs=args.epochs)
    
    if args.single:
        # 测试单个损失函数
        result = tester.test_loss_function(args.single)
        print(f"\n📊 测试结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        # 测试所有损失函数
        results_df = tester.run_all_tests()
        print(f"\n🎉 所有测试完成!")


if __name__ == "__main__":
    main() 