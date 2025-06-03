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
from training.data_loader import create_sample_data, KLineDataProcessor  # noqa: E402

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
            # 二分类模式：使用配置的类别权重
            binary_class_weights = torch.FloatTensor(
                self.config.class_weights
            ).to(self.device)
            self.criterion = BinaryClassificationLoss(
                self.config.task_weights, binary_class_weights
            )
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

        print("📊 模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: ~{total_params * 4 / 1024 / 1024:.1f}MB")
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
        """验证一个epoch - 使用简单二分类评估"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = {
            f'{h}min': [] for h in self.config.prediction_horizons
        }
        all_labels = {
            f'{h}min': [] for h in self.config.prediction_horizons
        }

        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(self.val_loader):
                # 数据移到设备
                features = features.to(self.device)
                batch_labels = {}
                for horizon, label_tensor in labels.items():
                    batch_labels[horizon] = label_tensor.squeeze().to(
                        self.device
                    )

                # 检查批次是否为空
                if features.size(0) == 0:
                    print(f"⚠️  警告: 批次 {batch_idx} 特征为空，跳过")
                    continue

                # 检查标签是否为空或尺寸不匹配
                skip_batch = False
                for horizon_key, label_tensor in batch_labels.items():
                    # 使用numel()检查张量是否为空，处理0维张量
                    if label_tensor.numel() == 0:
                        print(
                            f"⚠️  警告: 批次 {batch_idx} {horizon_key} "
                            f"标签为空，跳过此批次"
                        )
                        skip_batch = True
                        break
                    # 检查维度匹配
                    if (len(label_tensor.shape) == 0 or
                            label_tensor.shape[0] != features.shape[0]):
                        print(
                            f"⚠️  警告: 批次 {batch_idx} {horizon_key} "
                            f"尺寸不匹配"
                        )
                        print(
                            f"    特征形状: {features.shape}, "
                            f"标签形状: {label_tensor.shape}"
                        )
                        skip_batch = True
                        break

                if skip_batch:
                    continue

                # 前向传播
                predictions = self.model(features)

                # 额外的安全检查
                try:
                    losses = self.criterion(predictions, batch_labels)
                    total_loss += losses['total_loss'].item()
                except Exception as e:
                    print(f"❌ 批次 {batch_idx} 损失计算失败: {e}")
                    print(f"    特征形状: {features.shape}")
                    for k, v in batch_labels.items():
                        print(f"    {k} 标签形状: {v.shape}")
                    for k, v in predictions.items():
                        print(f"    {k} 预测形状: {v.shape}")
                    raise

                # 收集预测结果
                for horizon in self.config.prediction_horizons:
                    horizon_key = f'{horizon}min'
                    pred_labels = torch.argmax(predictions[horizon_key], dim=1)
                    all_predictions[horizon_key].extend(
                        pred_labels.cpu().numpy()
                    )
                    all_labels[horizon_key].extend(
                        batch_labels[horizon_key].cpu().numpy()
                    )

        # 计算简单二分类评估指标
        avg_loss = total_loss / len(self.val_loader)

        # 对每个时间尺度计算指标
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
        
        evaluation_results = {}
        for horizon in self.config.prediction_horizons:
            horizon_key = f'{horizon}min'
            y_true = np.array(all_labels[horizon_key])
            y_pred = np.array(all_predictions[horizon_key])

            # 计算基础分类指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            evaluation_results[horizon_key] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
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

        # 返回主要指标
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'f1_score': avg_f1,
            'detailed_results': evaluation_results
        }

        return metrics

    def train(self):
        """完整训练流程"""
        print("\n" + "="*50)
        print("🚀 模型训练阶段")
        print("="*50)

        # 早停相关变量
        best_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'f1_score': 0.0
        }
        patience_counter = 0

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

            # 打印训练结果
            epoch_time = time.time() - epoch_start_time
            print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_accuracy:.3f}")
            print(f"验证损失: {val_metrics['loss']:.4f}")

            # 打印评估结果
            print("\n🎯 稳定性检测评估:")
            print(f"  准确率: {val_metrics['accuracy']:.3f}")
            print(f"  精确率: {val_metrics['precision']:.3f}")
            print(f"  F1分数: {val_metrics['f1_score']:.3f}")

            print("各时间尺度表现:")
            for horizon, results in val_metrics['detailed_results'].items():
                print(
                    f"  {horizon}: "
                    f"准确率={results['accuracy']:.3f}, "
                    f"精确率={results['precision']:.3f}, "
                    f"召回率={results['recall']:.3f}, "
                    f"F1分数={results['f1_score']:.3f}"
                )

            print(f"耗时: {epoch_time:.2f}s")

            # 模型保存逻辑（基于准确率）
            if val_metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics['accuracy'] = val_metrics['accuracy']
                self.save_model('best_model.pth')
                print("💾 保存最佳准确率模型")
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停判断
            if patience_counter >= self.config.early_stopping_patience:
                print(f"🛑 早停触发: 连续{patience_counter}轮准确率未提升")
                break

            # 定期保存
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')

        print("\n✅ 训练完成！")
        print(f"最佳准确率: {best_metrics['accuracy']:.3f}")

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

            results[horizon_key] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'probabilities': y_proba
            }

            print(f"\n{horizon_key} 预测结果:")
            print(f"  准确率: {accuracy:.3f}")
            print(f"  精确率: {precision:.3f}")
            print(f"  召回率: {recall:.3f}")
            print(f"  F1分数: {f1:.3f}")
            print(f"  混淆矩阵:\n{cm}")

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
