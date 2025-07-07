#!/usr/bin/env python3
"""
🧪 深度学习实验主程序
独立实验程序，不影响现有工程

完整的K线数据深度学习预测实验
从数据处理到模型训练再到结果评估的端到端流程
"""

import argparse
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer import (  # noqa: E402
    BinaryClassificationLoss,
    MarketTransformer,
    MultiTaskLoss,
    TradingAwareLoss,
)

from training.config import ExperimentConfig  # noqa: E402
from training.data_loader import KLineDataProcessor, create_sample_data  # noqa: E402

warnings.filterwarnings('ignore')


class DeepLearningExperiment:
    """深度学习实验主控制器"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = self._get_device()
        # 初始化组件
        self.data_processor = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.writer = None

        # 训练状态
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

        print("🧪 实验初始化完成")
        print(f"📱 设备: {self.device}")
        print(config.summary())

    def _get_device(self) -> torch.device:
        """自动检测可用设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 使用GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("🖥️  使用CPU")
        return device

    def prepare_data(self):
        """准备训练数据"""
        print("\n" + "="*50)
        print("📊 数据准备阶段")
        print("="*50)

        # 检查数据文件是否存在
        if not os.path.exists(self.config.data_file):
            print(f"❌ 数据文件不存在: {self.config.data_file}")
            print("🧪 创建示例数据进行实验...")
            sample_path = create_sample_data()
            self.config.data_file = sample_path

        # 创建数据处理器
        config_dict = {
            'data_file': self.config.data_file,
            'sequence_length': self.config.sequence_length,
            'prediction_horizons': self.config.prediction_horizons,
            'price_change_threshold': self.config.price_change_threshold,
            'train_ratio': self.config.train_ratio,
            'val_ratio': self.config.val_ratio,
            'test_ratio': self.config.test_ratio,
            'batch_size': self.config.batch_size
        }

        self.data_processor = KLineDataProcessor(config_dict)

        # 加载原始数据
        df = self.data_processor.load_csv_data(self.config.data_file)

        # 特征工程和标签创建
        features, labels, feature_names = (
            self.data_processor.prepare_features_and_labels(df)
        )

        # 数据分割
        train_data, val_data, test_data = (
            self.data_processor.split_data(features, labels)
        )

        # 特征缩放
        self.data_processor.fit_scaler(train_data[0])
        train_features = self.data_processor.transform_features(train_data[0])
        val_features = self.data_processor.transform_features(val_data[0])
        test_features = self.data_processor.transform_features(test_data[0])

        train_data = (train_features, train_data[1])
        val_data = (val_features, val_data[1])
        test_data = (test_features, test_data[1])

        # 创建数据加载器
        (self.train_loader, self.val_loader,
         self.test_loader) = self.data_processor.create_data_loaders(
            train_data, val_data, test_data
        )

        # 保存预处理器
        preprocessor_path = os.path.join(
            self.config.output_dir, 'preprocessor.pkl'
        )
        self.data_processor.save_preprocessor(preprocessor_path)

        print("✅ 数据准备完成")

    def build_model(self):
        """构建模型"""
        print("\n" + "="*50)
        print("🏗️  模型构建阶段")
        print("="*50)

        # 模型配置
        model_config = {
            'feature_dim': self.config.feature_dim,
            'embed_dim': self.config.transformer_config['embed_dim'],
            'num_heads': self.config.transformer_config['num_heads'],
            'num_layers': self.config.transformer_config['num_layers'],
            'ff_dim': self.config.transformer_config['ff_dim'],
            'dropout': self.config.transformer_config['dropout'],
            'sequence_length': self.config.sequence_length,
            'prediction_horizons': self.config.prediction_horizons
        }

        # 创建模型
        self.model = MarketTransformer(model_config).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度器
        if self.config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_scheduler_params['T_max'],
                eta_min=self.config.lr_scheduler_params['eta_min']
            )

        # 损失函数
        if self.config.use_binary_classification:
            # 根据配置选择损失函数类型
            loss_type = self.config.loss_function_type
            task_weights = self.config.task_weights
            
            if loss_type == "binary_cross_entropy":
                # 标准二分类交叉熵（默认）
                binary_class_weights = torch.FloatTensor(
                    self.config.class_weights
                ).to(self.device)
                self.criterion = BinaryClassificationLoss(
                    task_weights, binary_class_weights
                )
                
            elif loss_type == "probability_adjusted":
                # 基础概率调整损失
                from models.transformer import ProbabilityAdjustedLoss
                params = self.config.loss_function_params["probability_adjusted"]
                self.criterion = ProbabilityAdjustedLoss(
                    task_weights,
                    base_stable_prob=params["base_stable_prob"],
                    base_change_prob=params["base_change_prob"]
                )
                
            elif loss_type == "confidence_weighted":
                # 置信度动态权重损失
                from models.transformer import ConfidenceWeightedLoss
                params = self.config.loss_function_params["confidence_weighted"]
                self.criterion = ConfidenceWeightedLoss(
                    task_weights,
                    confidence_threshold=params["confidence_threshold"],
                    high_conf_correct_weight=params["high_conf_correct_weight"],
                    high_conf_wrong_weight=params["high_conf_wrong_weight"],
                    low_conf_weight=params["low_conf_weight"]
                )
                
            elif loss_type == "business_cost":
                # 业务成本驱动损失
                from models.transformer import BusinessCostLoss
                params = self.config.loss_function_params["business_cost"]
                self.criterion = BusinessCostLoss(
                    task_weights,
                    false_alarm_cost=params["false_alarm_cost"],
                    miss_change_cost=params["miss_change_cost"],
                    stable_correct_reward=params["stable_correct_reward"],
                    change_correct_reward=params["change_correct_reward"]
                )
                
            elif loss_type == "imbalanced_focal":
                # 改进Focal Loss
                from models.transformer import ImbalancedFocalLoss
                params = self.config.loss_function_params["imbalanced_focal"]
                self.criterion = ImbalancedFocalLoss(
                    task_weights,
                    alpha=params["alpha"],
                    gamma=params["gamma"],
                    dynamic_adjustment=params["dynamic_adjustment"]
                )
                
            else:
                raise ValueError(f"未知的损失函数类型: {loss_type}")
                
            print(f"📊 使用损失函数: {loss_type}")
            
        else:
            # 三分类模式：使用原有逻辑
            class_weights = torch.FloatTensor(
                self.config.class_weights
            ).to(self.device)

            if self.config.use_trading_aware_loss:
                # 使用交易感知损失函数
                self.criterion = TradingAwareLoss(
                    self.config.task_weights,
                    self.config.direction_error_matrix,
                    class_weights
                )
            else:
                # 使用传统多任务损失函数
                self.criterion = MultiTaskLoss(
                    self.config.task_weights, class_weights
                )

        # TensorBoard
        self.writer = SummaryWriter(self.config.log_dir)

        # 模型统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"📊 模型参数统计: 总参数={total_params:,} | 可训练参数={trainable_params:,} | 模型大小=~{total_params * 4 / 1024 / 1024:.1f}MB")
        print("✅ 模型构建完成")

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = {
            f'{h}min': 0 for h in self.config.prediction_horizons
        }

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # 数据移到设备
            features = features.to(self.device)
            batch_labels = {}
            for horizon, label_tensor in labels.items():
                batch_labels[horizon] = label_tensor.squeeze().to(
                    self.device
                )

            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(features)

            # 计算损失
            losses = self.criterion(predictions, batch_labels)
            total_loss += losses['total_loss'].item()

            # 反向传播
            losses['total_loss'].backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.optimizer.step()

            # 计算准确率
            batch_size = features.size(0)
            total_samples += batch_size

            for horizon in self.config.prediction_horizons:
                horizon_key = f'{horizon}min'
                pred_labels = torch.argmax(predictions[horizon_key], dim=1)
                correct = (
                    pred_labels == batch_labels[horizon_key]
                ).sum().item()
                correct_predictions[horizon_key] += correct

            # 记录日志
            if batch_idx % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"  Batch {batch_idx:4d}/{len(self.train_loader)}: "
                    f"Loss={avg_loss:.4f}"
                )

        # 计算epoch统计
        avg_loss = total_loss / len(self.train_loader)
        accuracies = {
            k: v / total_samples for k, v in correct_predictions.items()
        }
        avg_accuracy = np.mean(list(accuracies.values()))

        return avg_loss, avg_accuracy, accuracies

    def validate_epoch(self):
        """验证一个epoch - 使用交易感知评估"""
        self.model.eval()
        total_loss = 0.0
        
        # 收集预测结果
        all_predictions = {f'{h}min': [] for h in self.config.prediction_horizons}
        all_labels = {f'{h}min': [] for h in self.config.prediction_horizons}
        all_probabilities = {f'{h}min': [] for h in self.config.prediction_horizons}

        with torch.no_grad():
            for features, batch_labels in self.val_loader:
                features = features.to(self.device)
                
                # 🔧 修复：确保标签也移动到正确设备
                labels_on_device = {}
                for horizon, label_tensor in batch_labels.items():
                    labels_on_device[horizon] = label_tensor.squeeze().to(self.device)
                
                # 计算损失
                predictions = self.model(features)
                batch_loss = self.criterion(predictions, labels_on_device)
                total_loss += batch_loss['total_loss'].item()

                # 获取预测概率
                probabilities = self.model.predict_proba(features)

                # 收集预测结果
                for horizon in self.config.prediction_horizons:
                    horizon_key = f'{horizon}min'
                    pred_labels = torch.argmax(predictions[horizon_key], dim=1)
                    all_predictions[horizon_key].extend(
                        pred_labels.cpu().numpy()
                    )
                    all_labels[horizon_key].extend(
                        labels_on_device[horizon_key].cpu().numpy()
                    )
                    all_probabilities[horizon_key].extend(
                        probabilities[horizon_key].cpu().numpy()
                    )

        # 计算交易感知评估指标
        avg_loss = total_loss / len(self.val_loader)

        # 对每个时间尺度计算指标
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        
        evaluation_results = {}
        for horizon in self.config.prediction_horizons:
            horizon_key = f'{horizon}min'
            y_true = np.array(all_labels[horizon_key])
            y_pred = np.array(all_predictions[horizon_key])
            y_proba = np.array(all_probabilities[horizon_key])

            # 计算基础分类指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # 计算混淆矩阵用于类别准确率
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            # 计算各类别准确率（召回率）
            stable_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0  # 稳定类准确率
            change_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0  # 变化类准确率
            
            # 🎯 计算灾难性错误率（方向完全相反）
            catastrophic_errors = 0  # 二分类中，两个类别互相误判都算灾难性错误
            total_samples = len(y_true)
            if total_samples > 0:
                false_positives = cm[0, 1]  # 稳定误判为变化
                false_negatives = cm[1, 0]  # 变化误判为稳定
                catastrophic_errors = false_positives + false_negatives
                catastrophic_error_rate = catastrophic_errors / total_samples
            else:
                catastrophic_error_rate = 0.0
            
            # 计算平衡类别准确率
            balanced_class_accuracy = (
                self.config.class_accuracy_weights['stable_weight'] * stable_accuracy +
                self.config.class_accuracy_weights['change_weight'] * change_accuracy
            )
            
            # 🎯 计算综合评分
            composite_score = self._calculate_composite_score(
                balanced_class_accuracy, catastrophic_error_rate
            )
            
            # 🎯 检测模型是否失败
            is_failed = self._check_model_failure(
                change_accuracy, stable_accuracy, balanced_class_accuracy, catastrophic_error_rate
            )
            
            evaluation_results[horizon_key] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'stable_accuracy': stable_accuracy,
                'change_accuracy': change_accuracy,
                'balanced_class_accuracy': balanced_class_accuracy,
                'catastrophic_error_rate': catastrophic_error_rate,
                'composite_score': composite_score,
                'is_failed_model': is_failed,
                'confusion_matrix': cm,
                'probabilities': y_proba
            }

        # 计算平均指标
        avg_accuracy = np.mean([
            results['accuracy'] for results in evaluation_results.values()
        ])
        avg_precision = np.mean([
            results['precision'] for results in evaluation_results.values()
        ])
        avg_f1 = np.mean([
            results['f1_score'] for results in evaluation_results.values()
        ])
        avg_balanced_class_accuracy = np.mean([
            results['balanced_class_accuracy'] for results in evaluation_results.values()
        ])
        avg_composite_score = np.mean([
            results['composite_score'] for results in evaluation_results.values()
        ])
        avg_catastrophic_rate = np.mean([
            results['catastrophic_error_rate'] for results in evaluation_results.values()
        ])

        # 返回主要指标
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'f1_score': avg_f1,
            'balanced_class_accuracy': avg_balanced_class_accuracy,
            'composite_score': avg_composite_score,
            'catastrophic_error_rate': avg_catastrophic_rate,
            'detailed_results': evaluation_results
        }

        return metrics

    def _calculate_composite_score(self, balanced_accuracy: float, 
                                 catastrophic_rate: float) -> float:
        """计算综合评分（去掉F1分数）"""
        weights = self.config.composite_score_weights
        
        # 归一化灾难性错误控制分数（错误率越低分数越高）
        catastrophic_control_score = max(0.0, 1.0 - catastrophic_rate * 20)  # 5%错误率对应0分
        
        composite_score = (
            weights['balanced_class_accuracy'] * balanced_accuracy +
            weights['catastrophic_control'] * catastrophic_control_score
        )
        
        return composite_score
    
    def _check_model_failure(self, change_accuracy: float, stable_accuracy: float,
                           balanced_accuracy: float, catastrophic_rate: float) -> bool:
        """检测模型是否失败"""
        criteria = self.config.model_failure_criteria
        
        return (
            change_accuracy < criteria['min_change_accuracy'] or
            stable_accuracy < criteria['min_stable_accuracy'] or
            balanced_accuracy < criteria['min_balanced_accuracy'] or
            catastrophic_rate > criteria['max_catastrophic_rate']
        )

    def train(self):
        """完整训练流程"""
        print("\n" + "="*50)
        print("🚀 模型训练阶段")
        print("="*50)

        # 早停相关变量
        best_metrics = {
            'composite_score': 0.0,
            'balanced_class_accuracy': 0.0,
            'change_accuracy': 0.0,
        }
        patience_counter = 0
        training_history = []

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            print(f"\n📈 Epoch {epoch+1}/{self.config.max_epochs}")
            print("-" * 40)

            # 训练
            train_loss, train_accuracy, train_accuracies = (
                self.train_epoch(epoch)
            )

            # 验证
            val_metrics = self.validate_epoch()

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_accuracy'].append(
                val_metrics['accuracy']
            )
            
            # 添加到训练历史（用于多指标早停）
            training_history.append(val_metrics)

            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar(
                'Loss/Validation', val_metrics['loss'], epoch
            )
            self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            self.writer.add_scalar(
                'Accuracy/Validation',
                val_metrics['accuracy'], epoch
            )
            
            # 记录新的关键指标
            self.writer.add_scalar('Metrics/Composite_Score', val_metrics['composite_score'], epoch)
            self.writer.add_scalar('Metrics/Balanced_Class_Accuracy', val_metrics['balanced_class_accuracy'], epoch)
            self.writer.add_scalar('Metrics/Catastrophic_Error_Rate', val_metrics['catastrophic_error_rate'], epoch)

            # 打印训练结果
            epoch_time = time.time() - epoch_start_time
            print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_accuracy:.3f}")
            print(f"验证损失: {val_metrics['loss']:.4f}")

            # 打印交易感知评估结果
            print(f"\n🎯 交易感知评估: 综合评分={val_metrics['composite_score']:.3f} | "
                  f"平衡准确率={val_metrics['balanced_class_accuracy']:.3f} | "
                  f"灾难错误率={val_metrics['catastrophic_error_rate']:.3f}")

            for horizon, results in val_metrics['detailed_results'].items():
                failed_status = "❌失败" if results['is_failed_model'] else "✅正常"
                cm = results['confusion_matrix']
                print(f"{horizon}: {failed_status} | 稳定类={results['stable_accuracy']:.3f}, 变化类={results['change_accuracy']:.3f}, 综合评分={results['composite_score']:.3f} | 混淆矩阵{cm.tolist()}")

            print(f"耗时: {epoch_time:.2f}s")

            # 多重模型保存逻辑
            model_improved = False
            
            # 1. 综合评分最佳模型
            if val_metrics['composite_score'] > best_metrics['composite_score']:
                best_metrics['composite_score'] = val_metrics['composite_score']
                self.save_model('best_composite.pth')
                print("💾 保存最佳综合评分模型")
                model_improved = True
                
            # 2. 平衡准确率最佳模型
            if val_metrics['balanced_class_accuracy'] > best_metrics['balanced_class_accuracy']:
                best_metrics['balanced_class_accuracy'] = val_metrics['balanced_class_accuracy']
                self.save_model('best_balanced.pth')
                print("💾 保存最佳平衡准确率模型")
                
            # 3. 变化类准确率最佳模型
            for horizon, results in val_metrics['detailed_results'].items():
                if results['change_accuracy'] > best_metrics['change_accuracy']:
                    best_metrics['change_accuracy'] = results['change_accuracy']
                    self.save_model('best_change.pth')
                    print("💾 保存最佳变化类准确率模型")

            # 🎯 改进的早停判断
            should_stop, stop_reason = self._should_stop_training(
                training_history, patience_counter
            )
            
            if model_improved:
                patience_counter = 0
            else:
                patience_counter += 1
                
            if should_stop:
                print(f"🛑 早停触发: {stop_reason}")
                break

            # 定期保存
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')

        print("\n✅ 训练完成！")
        print(f"最佳综合评分: {best_metrics['composite_score']:.3f}")
        print(f"最佳平衡准确率: {best_metrics['balanced_class_accuracy']:.3f}")
        print(f"最佳变化类准确率: {best_metrics['change_accuracy']:.3f}")

        # 输出所有轮训练的详细历史
        print("\n📈 训练历史回顾:")
        print("=" * 80)
        
        # 获取参数组合信息
        if (self.config.loss_function_type == "business_cost" and 
            "business_cost" in self.config.loss_function_params):
            bc_params = self.config.loss_function_params["business_cost"]
            false_alarm = bc_params.get("false_alarm_cost", "N/A")
            miss_change = bc_params.get("miss_change_cost", "N/A")
            stable_correct_reward = bc_params.get("stable_correct_reward", "N/A")
            change_correct_reward = bc_params.get("change_correct_reward", "N/A")
            param_info = f"参数[误报={false_alarm}, 漏报={miss_change}, 奖励=[稳定={stable_correct_reward}, 变化={change_correct_reward}]"
        else:
            param_info = "参数[标准配置]"
        
        print(f"配置: {param_info}")
        
        for epoch, metrics in enumerate(training_history, 1):
            for horizon, results in metrics['detailed_results'].items():
                failed_status = "❌失败" if results['is_failed_model'] else "✅正常"
                cm = results['confusion_matrix']
                print(f"Epoch{epoch} {horizon}: {failed_status} | 稳定类={results['stable_accuracy']:.3f}, 变化类={results['change_accuracy']:.3f}, 综合评分={results['composite_score']:.3f} | 混淆矩阵{cm.tolist()}")
        print("=" * 80)

    def _should_stop_training(self, history: list, patience_counter: int) -> tuple:
        """改进的早停判断逻辑"""
        if len(history) < 5:  # 至少训练5轮
            return False, ""
            
        # 基础耐心检查
        if patience_counter >= self.config.early_stopping_patience:
            return True, f"连续{patience_counter}轮综合评分未提升"
        
        # 多指标早停检查
        if self.config.multi_metric_early_stopping['enabled']:
            window = self.config.multi_metric_early_stopping['stable_trend_window']
            if len(history) >= window:
                recent_metrics = history[-window:]
                
                # 检查变化类准确率是否严重下降
                change_accuracies = []
                for metrics in recent_metrics:
                    for results in metrics['detailed_results'].values():
                        change_accuracies.append(results['change_accuracy'])
                
                if len(change_accuracies) >= 2:
                    change_trend = change_accuracies[-1] - change_accuracies[0]
                    decline_limit = self.config.multi_metric_early_stopping['change_accuracy_decline_limit']
                    
                    if change_trend < decline_limit:
                        return True, f"变化类准确率下降过多: {change_trend:.3f}"
                
                # 检查综合评分是否停滞
                composite_scores = [m['composite_score'] for m in recent_metrics]
                if len(composite_scores) >= 2:
                    score_improvement = max(composite_scores) - min(composite_scores)
                    min_improvement = self.config.multi_metric_early_stopping['min_improvement_threshold']
                    
                    if score_improvement < min_improvement and patience_counter >= 8:
                        return True, f"综合评分改善停滞: {score_improvement:.4f} < {min_improvement}"
        
        return False, ""

    def evaluate(self):
        """在测试集上评估"""
        print("\n" + "="*50)
        print("📊 模型评估阶段")
        print("="*50)

        # 加载最佳模型
        best_model_path = os.path.join(
            self.config.model_save_dir, 'best_model.pth'
        )
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(
                best_model_path, map_location=self.device
            ))
            print("📂 加载最佳模型")

        self.model.eval()
        all_predictions = {
            f'{h}min': [] for h in self.config.prediction_horizons
        }
        all_labels = {
            f'{h}min': [] for h in self.config.prediction_horizons
        }
        all_probabilities = {
            f'{h}min': [] for h in self.config.prediction_horizons
        }

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)

                # 预测
                predictions = self.model(features)
                probabilities = self.model.predict_proba(features)

                for horizon in self.config.prediction_horizons:
                    horizon_key = f'{horizon}min'

                    # 预测标签
                    pred_labels = torch.argmax(
                        predictions[horizon_key], dim=1
                    )
                    all_predictions[horizon_key].extend(
                        pred_labels.cpu().numpy()
                    )

                    # 真实标签
                    true_labels = labels[horizon_key].squeeze()
                    all_labels[horizon_key].extend(true_labels.numpy())

                    # 预测概率
                    probs = probabilities[horizon_key]
                    all_probabilities[horizon_key].extend(
                        probs.cpu().numpy()
                    )

        # 计算评估指标
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )

        results = {}
        for horizon in self.config.prediction_horizons:
            horizon_key = f'{horizon}min'

            y_true = np.array(all_labels[horizon_key])
            y_pred = np.array(all_predictions[horizon_key])
            y_proba = np.array(all_probabilities[horizon_key])

            # 基础指标
            accuracy = accuracy_score(y_true, y_pred)
            (precision, recall,
             f1, _) = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            cm = confusion_matrix(y_true, y_pred)

            # 计算各类别准确率
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            stable_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 稳定类准确率
            change_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 变化类准确率

            results[horizon_key] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'probabilities': y_proba,
                'stable_accuracy': stable_accuracy,
                'change_accuracy': change_accuracy,
            }

            print(f"\n{horizon_key} 预测结果: 准确率={accuracy:.3f} | "
                  f"精确率={precision:.3f} | 召回率={recall:.3f} | F1={f1:.3f}")
            print(f"  混淆矩阵: {cm.tolist()}")
            print(f"  稳定类准确率: {stable_accuracy:.3f} | 变化类准确率: {change_accuracy:.3f}")

        return results

    def save_model(self, filename: str):
        """保存模型"""
        model_path = os.path.join(self.config.model_save_dir, filename)
        torch.save(self.model.state_dict(), model_path)

    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.config.plot_training_curves:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.training_history['train_loss'], label='训练损失')
        axes[0, 0].plot(self.training_history['val_loss'], label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 准确率曲线
        axes[0, 1].plot(self.training_history['train_accuracy'], label='训练准确率')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='验证准确率')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.config.plot_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 训练曲线已保存: {plot_path}")

    def run_experiment(self):
        """运行完整实验"""
        try:
            print("🧪 开始深度学习实验")
            print("=" * 60)

            # 1. 数据准备
            self.prepare_data()

            # 2. 模型构建
            self.build_model()

            # 3. 模型训练
            self.train()

            # 4. 模型评估
            results = self.evaluate()

            # 5. 可视化
            self.plot_training_curves()

            print("\n" + "=" * 60)
            print("🎉 实验完成！")
            print("=" * 60)

            return results

        except Exception as e:
            print(f"\n❌ 实验失败: {e}")
            raise
        finally:
            if self.writer:
                self.writer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='深度学习实验程序')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data', type=str, help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')

    args = parser.parse_args()

    # 创建配置
    config = ExperimentConfig()

    # 命令行参数覆盖
    if args.data:
        config.data_file = args.data
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    # 运行实验
    experiment = DeepLearningExperiment(config)
    results = experiment.run_experiment()

    return results


if __name__ == "__main__":
    main()
    main()
