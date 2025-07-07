#!/usr/bin/env python3
"""
业务成本损失函数测试版本
支持27种参数配置的串行测试
现在支持灵活的命令行参数控制
"""

import argparse
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="BusinessCost损失函数参数优化测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行全部81种组合
  python %(prog)s
  
  # 运行特定组合
  python %(prog)s --single 2 7 0.3 0.4
  
  # 运行参数范围
  python %(prog)s --range --false-alarm 1,2 --miss-change 7,8 --stable-reward 0.3 --change-reward 0.4
  
  # 运行前5个组合
  python %(prog)s --count 5
  
  # 运行指定编号的组合
  python %(prog)s --indices 5,14,22
  
  # 只分析已有结果
  python %(prog)s --analyze-only
        """
    )
    
    # 参数组合控制
    group_combo = parser.add_argument_group('参数组合控制')
    group_combo.add_argument('--single', nargs=4, metavar=('F', 'M', 'SR', 'CR'),
                             help='运行单个组合 (误报成本, 漏报成本, 稳定奖励, 变化奖励)')
    group_combo.add_argument('--range', action='store_true',
                             help='使用范围模式 (需配合参数范围选项)')
    group_combo.add_argument('--false-alarm', type=str,
                             help='误报成本范围，逗号分隔 (如: 1,2,3)')
    group_combo.add_argument('--miss-change', type=str,
                             help='漏报成本范围，逗号分隔 (如: 6,7,8)')
    group_combo.add_argument('--stable-reward', type=str,
                             help='稳定奖励范围，逗号分隔 (如: 0.2,0.3,0.4)')
    group_combo.add_argument('--change-reward', type=str,
                             help='变化奖励范围，逗号分隔 (如: 0.2,0.3,0.4)')
    group_combo.add_argument('--count', type=int,
                             help='只运行前N个组合')
    group_combo.add_argument('--start', type=int, default=1,
                             help='从第N个组合开始 (默认: 1)')
    group_combo.add_argument('--indices', type=str,
                             help='运行指定编号的组合，逗号分隔 (如: 1,5,10)')
    
    # 结果管理
    group_result = parser.add_argument_group('结果管理')
    group_result.add_argument('--analyze-only', action='store_true',
                              help='只分析已有结果，不重新训练')
    group_result.add_argument('--result-file', type=str,
                              help='指定结果文件路径')
    group_result.add_argument('--resume', action='store_true',
                              help='继续中断的测试')
    group_result.add_argument('--retry-failed', action='store_true',
                              help='重跑失败的组合')
    
    # 实用工具
    group_util = parser.add_argument_group('实用工具')
    group_util.add_argument('--list-combinations', action='store_true',
                            help='列出所有组合编号')
    group_util.add_argument('--validate', nargs=3, metavar=('F', 'M', 'R'),
                            help='验证参数组合有效性')
    
    return parser.parse_args()


def parse_param_list(param_str, param_type=float):
    """解析参数列表字符串"""
    if not param_str:
        return None
    try:
        return [param_type(x.strip()) for x in param_str.split(',')]
    except ValueError as e:
        raise ValueError(f"参数解析错误: {param_str} - {e}")


def validate_single_combination(false_alarm, miss_change, stable_reward, change_reward):
    """验证单个参数组合的有效性"""
    try:
        false_alarm = float(false_alarm)
        miss_change = float(miss_change)
        stable_reward = float(stable_reward)
        change_reward = float(change_reward)
        
        # 基本范围检查
        if false_alarm <= 0:
            return False, "误报成本必须大于0"
        if miss_change <= 0:
            return False, "漏报成本必须大于0"
        if stable_reward <= 0:
            return False, "稳定奖励必须大于0"
        if change_reward <= 0:
            return False, "变化奖励必须大于0"
        if stable_reward >= 1:
            return False, "稳定奖励应该小于1"
        if change_reward >= 1:
            return False, "变化奖励应该小于1"
            
        return True, "参数组合有效"
    except ValueError:
        return False, "参数类型错误，请使用数字"


def run_single_test_worker(params):
    """单个参数组合的工作进程"""
    false_alarm, miss_change, stable_reward, change_reward = params
    
    # 打印进程ID以验证多进程运行
    print(f"💼 工作进程 PID: {os.getpid()} 正在处理参数: {params}")
    
    print(f"\n🔬 进程启动: 误报={false_alarm}, 漏报={miss_change}, "
          f"稳定奖励={stable_reward}, 变化奖励={change_reward}")
    
    # 创建配置
    config = ExperimentConfig()
    config.loss_function_type = "business_cost"
    
    # 设置business_cost参数
    config.loss_function_params["business_cost"] = {
        "false_alarm_cost": false_alarm,
        "miss_change_cost": miss_change,
        "stable_correct_reward": stable_reward,
        "change_correct_reward": change_reward
    }
    
    # 输出目录配置
    param_name = (
        f"business_cost_f{false_alarm}_m{miss_change}_"
        f"sr{stable_reward}_cr{change_reward}"
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
        param_info = (
            f"配置: 参数[误报={false_alarm}, 漏报={miss_change}, "
            f"稳定奖励={stable_reward}, 变化奖励={change_reward}]"
        )
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
            'stable_correct_reward': stable_reward,
            'change_correct_reward': change_reward,
            'param_signature': (
                f"f{false_alarm}_m{miss_change}_sr{stable_reward}_cr{change_reward}"
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
            'stable_correct_reward': stable_reward,
            'change_correct_reward': change_reward,
            'param_signature': (
                f"f{false_alarm}_m{miss_change}_sr{stable_reward}_cr{change_reward}"
            ),
            'status': 'failed',
            'error': str(e),
            'change_accuracy': 0.0,
            'stable_accuracy': 0.0,
            'weighted_score': 0.0
        }


class BusinessCostTest:
    """业务成本参数测试类（串行版本）"""
    
    def __init__(self, custom_params=None):
        # 默认参数范围
        self.default_false_alarm_costs = [1, 2, 3]
        self.default_miss_change_costs = [6, 7, 8] 
        self.default_stable_rewards = [0.2, 0.3, 0.4]
        self.default_change_rewards = [0.2, 0.3, 0.4]
        
        # 根据输入设置实际使用的参数
        if custom_params:
            self.false_alarm_costs = custom_params.get('false_alarm', self.default_false_alarm_costs)
            self.miss_change_costs = custom_params.get('miss_change', self.default_miss_change_costs)
            self.stable_rewards = custom_params.get('stable_reward', self.default_stable_rewards)
            self.change_rewards = custom_params.get('change_reward', self.default_change_rewards)
        else:
            self.false_alarm_costs = self.default_false_alarm_costs
            self.miss_change_costs = self.default_miss_change_costs
            self.stable_rewards = self.default_stable_rewards
            self.change_rewards = self.default_change_rewards
        
        # 生成所有组合
        self.param_combinations = list(product(
            self.false_alarm_costs,
            self.miss_change_costs, 
            self.stable_rewards,
            self.change_rewards
        ))
        
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🧪 准备串行测试 {len(self.param_combinations)} 种参数组合")
        
    def list_all_combinations(self):
        """列出所有参数组合"""
        print("\n📋 所有参数组合列表:")
        print("=" * 80)
        for i, (f, m, sr, cr) in enumerate(self.param_combinations, 1):
            print(f"{i:2d}. 误报成本={f}, 漏报成本={m}, 稳定奖励={sr}, 变化奖励={cr}")
        print("=" * 80)
        
    def run_specific_combinations(self, indices):
        """运行指定编号的组合"""
        if not indices:
            return
            
        print(f"🎯 运行指定的 {len(indices)} 个组合")
        
        selected_combinations = []
        for idx in indices:
            if 1 <= idx <= len(self.param_combinations):
                selected_combinations.append(self.param_combinations[idx-1])
                print(f"  {idx}. {self.param_combinations[idx-1]}")
            else:
                print(f"⚠️  编号 {idx} 超出范围 (1-{len(self.param_combinations)})")
        
        if not selected_combinations:
            print("❌ 没有有效的组合可运行")
            return
            
        self._run_combinations(selected_combinations)
    
    def run_range_combinations(self, start=1, count=None):
        """运行范围内的组合"""
        start_idx = start - 1  # 转换为0-based索引
        if start_idx < 0:
            start_idx = 0
            
        if count:
            end_idx = min(start_idx + count, len(self.param_combinations))
            selected_combinations = self.param_combinations[start_idx:end_idx]
            print(f"🎯 运行第 {start} 到第 {start + len(selected_combinations) - 1} 个组合")
        else:
            selected_combinations = self.param_combinations[start_idx:]
            print(f"🎯 从第 {start} 个组合开始运行到结束")
            
        self._run_combinations(selected_combinations)
    
    def run_single_combination(self, false_alarm, miss_change, stable_reward, change_reward):
        """运行单个组合"""
        combination = (float(false_alarm), float(miss_change), float(stable_reward), float(change_reward))
        print(f"🎯 运行单个组合: {combination}")
        self._run_combinations([combination])
    
    def _run_combinations(self, combinations):
        """内部方法：运行指定的组合列表"""
        print("🚀 开始BusinessCost参数测试")
        print("=" * 80)
        
        start_time = time.time()
        
        for i, params in enumerate(combinations, 1):
            progress_percent = (i / len(combinations)) * 100
            print(f"\n📊 进度: {i}/{len(combinations)} "
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
                    'stable_correct_reward': params[2],
                    'change_correct_reward': params[3],
                    'param_signature': (
                        f"f{params[0]}_m{params[1]}_sr{params[2]}_cr{params[3]}"
                    ),
                    'status': 'failed',
                    'error': str(e),
                    'change_accuracy': 0.0,
                    'stable_accuracy': 0.0,
                    'weighted_score': 0.0
                }
                self.results.append(failed_result)
        
        total_time = time.time() - start_time
        print(f"\n🎉 测试完成！总耗时: {total_time/60:.1f}分钟")

    def run_all_tests(self):
        """运行所有参数组合测试（串行版本）"""
        self._run_combinations(self.param_combinations)
        
    def load_existing_results(self, result_file=None):
        """加载已有的测试结果"""
        if result_file and os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                print(f"✅ 成功加载结果文件: {result_file}")
                print(f"📊 加载了 {len(self.results)} 个结果")
                return True
            except Exception as e:
                print(f"❌ 加载结果文件失败: {e}")
                return False
        else:
            # 自动寻找最新的结果文件
            output_dir = "outputs"
            if os.path.exists(output_dir):
                json_files = [f for f in os.listdir(output_dir) 
                            if f.startswith('business_cost_test_') and f.endswith('.json')]
                if json_files:
                    latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
                    return self.load_existing_results(os.path.join(output_dir, latest_file))
            
            print("❌ 没有找到已有的结果文件")
            return False

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
            stable_reward = row['stable_correct_reward']
            change_reward = row['change_correct_reward']
            
            print(f"\n{i:2d}. 配置: 参数[误报={false_alarm}, 漏报={miss_change}, 稳定奖励={stable_reward}, 变化奖励={change_reward}]")
            
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
    args = parse_arguments()
    
    # 实用工具命令
    if args.validate:
        is_valid, message = validate_single_combination(*args.validate)
        print(f"验证结果: {message}")
        if is_valid:
            print("✅ 参数组合有效")
            sys.exit(0)
        else:
            print("❌ 参数组合无效")
            sys.exit(1)
    
    # 创建测试实例
    custom_params = None
    if args.range:
        custom_params = {}
        if args.false_alarm:
            custom_params['false_alarm'] = parse_param_list(args.false_alarm, float)
        if args.miss_change:
            custom_params['miss_change'] = parse_param_list(args.miss_change, float)
        if args.stable_reward:
            custom_params['stable_reward'] = parse_param_list(args.stable_reward, float)
        if args.change_reward:
            custom_params['change_reward'] = parse_param_list(args.change_reward, float)
    
    tester = BusinessCostTest(custom_params)
    
    # 列出组合命令
    if args.list_combinations:
        tester.list_all_combinations()
        return
    
    # 只分析模式
    if args.analyze_only:
        if tester.load_existing_results(args.result_file):
            tester.generate_report()
        return
    
    print("🚀 BusinessCost损失函数参数优化测试")
    if custom_params:
        print("自定义参数范围:")
        if 'false_alarm' in custom_params:
            print(f"  - 误报成本: {custom_params['false_alarm']}")
        if 'miss_change' in custom_params:
            print(f"  - 漏报成本: {custom_params['miss_change']}")
        if 'stable_reward' in custom_params:
            print(f"  - 稳定奖励: {custom_params['stable_reward']}")
        if 'change_reward' in custom_params:
            print(f"  - 变化奖励: {custom_params['change_reward']}")
    else:
        print("默认参数范围:")
        print("  - 误报成本: [1, 2, 3]")
        print("  - 漏报成本: [6, 7, 8]") 
        print("  - 稳定奖励: [0.2, 0.3, 0.4]")
        print("  - 变化奖励: [0.2, 0.3, 0.4]")
    
    print(f"  - 总组合数: {len(tester.param_combinations)}种")
    print("  - 执行方式: 串行")
    print("=" * 60)
    
    # 根据参数选择运行模式
    if args.single:
        # 验证单个组合参数
        is_valid, message = validate_single_combination(*args.single)
        if not is_valid:
            print(f"❌ {message}")
            sys.exit(1)
        tester.run_single_combination(*args.single)
    elif args.indices:
        try:
            indices = [int(x.strip()) for x in args.indices.split(',')]
            tester.run_specific_combinations(indices)
        except ValueError:
            print("❌ 编号格式错误，请使用逗号分隔的数字 (如: 1,5,10)")
            sys.exit(1)
    elif args.count or args.start > 1:
        tester.run_range_combinations(args.start, args.count)
    else:
        # 默认运行所有组合
        tester.run_all_tests()
    
    # 生成分析报告
    tester.generate_report()
    
    print("\n🎉 BusinessCost参数优化测试完成！")


if __name__ == "__main__":
    main() 