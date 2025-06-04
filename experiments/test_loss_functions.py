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
        base_output_dir = "outputs"  # 直接使用outputs，因为我们在experiments目录下
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
        print(f"\n📈 权重配置对比测试结果:")
        print("="*80)
        
        successful_tests = df[df['status'] == 'success']
        
        if len(successful_tests) > 0:
            # 🎯 为每个权重配置单独输出详细结果
            print("📊 各权重配置详细结果:")
            print("-"*80)
            
            for i, (_, row) in enumerate(successful_tests.iterrows(), 1):
                weight = row.get('weight_config', 'N/A')
                desc = row.get('description', row['loss_function'])
                
                print(f"\n🔸 配置 {i}: {desc}")
                print(f"   权重设置: [1.0, {weight}]")
                
                # 核心性能指标
                stable_acc = row.get('stable_accuracy', 0)
                change_acc = row.get('change_accuracy', 0)
                f1 = row.get('f1_score', 0)
                accuracy = row.get('accuracy', 0)
                
                # 状态评估
                if row.get('is_failed_model', False):
                    status = "❌ 失败"
                elif change_acc > 0.5 and stable_acc > 0.6:
                    status = "✅ 优秀"
                elif change_acc > 0.3:
                    status = "⚠️  一般"  
                else:
                    status = "🔴 较差"
                
                print(f"   评估状态: {status}")
                print(f"   整体准确率: {accuracy:.3f}")
                print(f"   F1分数: {f1:.3f}")
                print(f"   稳定类准确率: {stable_acc:.3f}")
                print(f"   变化类准确率: {change_acc:.3f}")
                
                # 混淆矩阵
                cm = row.get('confusion_matrix', None)
                if cm:
                    print(f"   混淆矩阵:")
                    print(f"             预测")
                    print(f"   实际   稳定(0)  变化(1)")
                    print(f"   稳定(0)  {cm[0][0]:6d}   {cm[0][1]:6d}")
                    print(f"   变化(1)  {cm[1][0]:6d}   {cm[1][1]:6d}")
                
                # 错误分析
                fp = row.get('false_positives', 0)
                fn = row.get('false_negatives', 0) 
                catastrophic_rate = row.get('catastrophic_error_rate', 0)
                
                print(f"   误报数(稳定→变化): {fp}")
                print(f"   漏报数(变化→稳定): {fn}")
                print(f"   灾难错误率: {catastrophic_rate:.3f}")
                
                # 综合评分
                if 'composite_score' in row:
                    print(f"   综合评分: {row['composite_score']:.3f}")
                if 'balanced_class_accuracy' in row:
                    print(f"   平衡准确率: {row['balanced_class_accuracy']:.3f}")
                
                # 模型路径
                model_path = row.get('model_save_dir', 'N/A')
                print(f"   模型路径: {model_path}")
                print()
            
            # 🎯 权重配置对比总结
            print("="*80)
            print("📈 权重效果对比总结:")
            print("-"*50)
            
            # 按变化类准确率排序
            sorted_by_change = successful_tests.sort_values('change_accuracy', ascending=False)
            
            print("🎯 变化类准确率排名:")
            for i, (_, row) in enumerate(sorted_by_change.iterrows(), 1):
                weight = row.get('weight_config', 'N/A')
                change_acc = row.get('change_accuracy', 0)
                stable_acc = row.get('stable_accuracy', 0)
                
                balance_indicator = "⚖️ 平衡" if abs(change_acc - stable_acc) < 0.3 else "⚠️ 不平衡"
                print(f"  {i}. 权重[1.0, {weight}]: 变化类{change_acc:.3f} | 稳定类{stable_acc:.3f} {balance_indicator}")
            
            # 数学期望分析
            print(f"\n🧮 数学期望分析 (93.7%稳定类, 6.3%变化类):")
            for _, row in successful_tests.iterrows():
                weight = row.get('weight_config', 'N/A')
                if weight != 'N/A':
                    expected_weight = 0.937 * 1.0 + 0.063 * weight
                    change_acc = row.get('change_accuracy', 0)
                    print(f"  权重[1.0, {weight}]: 期望权重={expected_weight:.3f}, 实际变化类准确率={change_acc:.3f}")
        
        # 失败测试
        failed_tests = df[df['status'] == 'failed']
        if len(failed_tests) > 0:
            print(f"\n❌ 运行失败的测试:")
            for _, row in failed_tests.iterrows():
                print(f"  {row['loss_function']}: {row.get('error', 'Unknown error')}")

        # 🎯 推荐结论
        if len(successful_tests) > 0:
            print(f"\n🎯 权重调整建议:")
            
            # 分析权重趋势
            weights_performance = []
            for _, row in successful_tests.iterrows():
                weight = row.get('weight_config', None)
                change_acc = row.get('change_accuracy', 0)
                if weight is not None:
                    weights_performance.append((weight, change_acc))
            
            if len(weights_performance) >= 2:
                weights_performance.sort()
                best_weight, best_change_acc = max(weights_performance, key=lambda x: x[1])
                
                print(f"  📊 最佳权重: [1.0, {best_weight}]")
                print(f"  📈 最佳变化类准确率: {best_change_acc:.3f}")
                
                if best_change_acc < 0.3:
                    print(f"  💡 建议: 变化类准确率仍较低，考虑:")
                    print(f"     - 进一步提升权重至 [1.0, {best_weight + 2}] 或 [1.0, {best_weight + 5}]")
                    print(f"     - 尝试其他损失函数 (business_cost, focal_loss)")
                    print(f"     - 调整学习率或训练策略")
                elif best_change_acc > 0.6:
                    print(f"  ✅ 权重调整有效！可尝试微调至 [1.0, {best_weight + 0.5}] 优化平衡性")
                else:
                    print(f"  ⚠️  有进展但仍需改进，建议继续调整权重")
            else:
                print("  ❌ 没有足够的数据进行权重趋势分析")


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