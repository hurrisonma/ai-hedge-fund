"""
🧪 Transformer模型架构
独立实验程序，专为K线时序预测设计
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # 线性变换和重塑
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑和输出变换
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.W_o(attention_output)
        return output, attention_weights

class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """单个Transformer块"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class MarketTransformer(nn.Module):
    """
    市场时序预测Transformer模型
    
    专为K线数据设计的Transformer架构，支持多时间尺度预测
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 配置参数
        self.feature_dim = config['feature_dim']
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.ff_dim = config['ff_dim']
        self.dropout = config['dropout']
        self.sequence_length = config['sequence_length']
        self.prediction_horizons = config['prediction_horizons']
        # 支持二分类和三分类
        self.num_classes = config.get('num_classes', 2)  # 默认为2（二分类）
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(self.feature_dim, self.embed_dim)
        self.input_norm = nn.LayerNorm(self.embed_dim)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.sequence_length)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 预测头（每个时间尺度一个）
        self.prediction_heads = nn.ModuleDict()
        for horizon in self.prediction_horizons:
            self.prediction_heads[f'{horizon}min'] = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
                nn.ReLU(), 
                nn.Dropout(self.dropout),
                nn.Linear(self.embed_dim // 4, self.num_classes)
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """创建padding mask"""
        if lengths is None:
            # 如果没有提供长度，假设所有序列都是完整的
            return None
        
        batch_size, seq_len = x.size(0), x.size(1)
        mask = torch.arange(seq_len, device=x.device).expand(
            batch_size, seq_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, sequence_length, feature_dim]
            lengths: 序列实际长度 [batch_size] (可选)
            
        Returns:
            每个时间尺度的预测结果字典
        """
        batch_size, seq_len, feature_dim = x.size()
        
        # 输入嵌入和归一化
        x = self.input_embedding(x)  # [batch, seq_len, embed_dim]
        x = self.input_norm(x)
        
        # 位置编码
        x = x.transpose(0, 1)  # [seq_len, batch, embed_dim]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, embed_dim]
        
        # 创建mask
        mask = self.create_padding_mask(x, lengths)
        
        # Transformer层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # 全局池化：取最后一个时间步的输出
        # 或者可以使用平均池化：x.mean(dim=1)
        pooled_output = x[:, -1, :]  # [batch, embed_dim]
        
        # 多任务预测
        predictions = {}
        for horizon in self.prediction_horizons:
            pred_logits = self.prediction_heads[f'{horizon}min'](pooled_output)
            predictions[f'{horizon}min'] = pred_logits
        
        return predictions
    
    def predict_proba(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """预测概率分布"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probabilities = {}
            for horizon, logit in logits.items():
                probabilities[horizon] = F.softmax(logit, dim=-1)
        return probabilities
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """获取注意力权重（用于可视化）"""
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, feature_dim = x.size()
            
            # 前向传播到指定层
            x = self.input_embedding(x)
            x = self.input_norm(x)
            x = x.transpose(0, 1)
            x = self.positional_encoding(x)
            x = x.transpose(0, 1)
            
            target_layer = self.transformer_blocks[layer_idx]
            attention_output, attention_weights = target_layer.attention(x, x, x)
            
            return attention_weights

class BinaryClassificationLoss(nn.Module):
    """简化的二分类损失函数 - 专为涨跌预测设计"""
    
    def __init__(self, task_weights: Dict[str, float], 
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.task_weights = task_weights
        self.class_weights = class_weights
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算二分类损失
        
        Args:
            predictions: 模型预测结果 {horizon: [batch_size, 2]}
            targets: 真实标签 {horizon: [batch_size]} (0=涨, 1=跌)
            
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        for task_name, pred_logits in predictions.items():
            if task_name not in targets:
                continue
                
            target_labels = targets[task_name]
            
            # 🔧 输入验证
            if pred_logits.size(0) != target_labels.size(0):
                raise ValueError(f"预测和标签批次大小不匹配: {pred_logits.size(0)} vs {target_labels.size(0)}")
            
            if pred_logits.size(1) != 2:
                raise ValueError(f"二分类要求预测维度为2，但得到{pred_logits.size(1)}")
            
            # 检查标签值域（应该只有0和1）
            unique_labels = torch.unique(target_labels)
            if not all(label.item() in [0, 1] for label in unique_labels):
                raise ValueError(f"二分类标签应该只包含0和1，但得到{unique_labels.tolist()}")
            
            # 计算二分类交叉熵损失
            task_loss = F.cross_entropy(pred_logits, target_labels, 
                                      weight=self.class_weights)
            
            # 应用任务权重
            task_weight = self.task_weights.get(task_name, 1.0)
            weighted_loss = task_weight * task_loss
            
            losses[f'loss_{task_name}'] = task_loss
            total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses

class TradingAwareLoss(nn.Module):
    """交易感知损失函数 - 重点惩罚方向性错误"""
    
    def __init__(self, task_weights: Dict[str, float], 
                 direction_error_matrix: List[List[float]], 
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.task_weights = task_weights
        self.class_weights = class_weights
        
        # 转换错误权重矩阵为tensor
        self.error_matrix = torch.tensor(direction_error_matrix, dtype=torch.float32)
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算交易感知损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        for task_name, pred_logits in predictions.items():
            if task_name not in targets:
                continue
                
            target_labels = targets[task_name]
            
            # 基础交叉熵损失
            base_loss = F.cross_entropy(pred_logits, target_labels, 
                                      weight=self.class_weights, reduction='none')
            
            # 反持平偏好：额外惩罚预测持平类
            pred_probs = F.softmax(pred_logits, dim=1)
            flat_penalty = pred_probs[:, 2] * 0.5  # 对持平预测概率额外惩罚
            
            # 方向性奖励：奖励涨跌预测
            directional_reward = (pred_probs[:, 0] + pred_probs[:, 1]) * 0.2
            
            # 综合损失
            adjusted_loss = base_loss + flat_penalty - directional_reward
            
            # 计算方向性错误惩罚
            pred_classes = torch.argmax(pred_logits, dim=1)
            
            # 确保错误权重矩阵在正确的设备上
            error_matrix = self.error_matrix.to(pred_logits.device)
            
            # 获取每个样本的错误权重
            error_weights = error_matrix[target_labels, pred_classes]
            
            # 应用方向性错误惩罚
            directional_loss = adjusted_loss * (1.0 + error_weights * 0.1)
            task_loss = directional_loss.mean()
            
            # 应用任务权重
            task_weight = self.task_weights.get(task_name, 1.0)
            weighted_loss = task_weight * task_loss
            
            losses[f'loss_{task_name}'] = task_loss
            total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses

class MultiTaskLoss(nn.Module):
    """传统多任务损失函数（保留用于对比）"""
    
    def __init__(self, task_weights: Dict[str, float], class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.task_weights = task_weights
        self.class_weights = class_weights
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        for task_name, pred_logits in predictions.items():
            if task_name not in targets:
                continue
                
            target_labels = targets[task_name]
            
            # 计算交叉熵损失
            task_loss = F.cross_entropy(pred_logits, target_labels, weight=self.class_weights)
            
            # 应用任务权重
            task_weight = self.task_weights.get(task_name, 1.0)
            weighted_loss = task_weight * task_loss
            
            losses[f'loss_{task_name}'] = task_loss
            total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses

def create_model(config: Dict) -> MarketTransformer:
    """
    创建模型工厂函数
    
    Args:
        config: 模型配置字典
        
    Returns:
        初始化的模型
    """
    return MarketTransformer(config)

def test_transformer_model():
    """测试Transformer模型"""
    print("🧪 测试Transformer模型...")
    
    # 模拟配置
    config = {
        'feature_dim': 38,
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'ff_dim': 1024,
        'dropout': 0.1,
        'sequence_length': 30,
        'prediction_horizons': [5, 10, 15]
    }
    
    # 创建模型
    model = MarketTransformer(config)
    
    # 模拟输入数据
    batch_size = 32
    seq_length = 30
    feature_dim = 38
    
    x = torch.randn(batch_size, seq_length, feature_dim)
    
    # 前向传播
    outputs = model(x)
    
    print(f"输入形状: {x.shape}")
    print("输出形状:")
    for horizon, output in outputs.items():
        print(f"  {horizon}: {output.shape}")
    
    # 测试概率预测
    probas = model.predict_proba(x)
    print("\n概率预测:")
    for horizon, proba in probas.items():
        print(f"  {horizon}: {proba.shape}, sum={proba.sum(dim=1)[0]:.3f}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return model

class TwoTaskTransformer(nn.Module):
    """
    两任务Transformer模型 - 稳定性检测 + 方向判断
    
    任务1：稳定性检测 (stable vs unstable)
    任务2：方向判断 (up vs down, 仅在unstable时)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 配置参数
        self.feature_dim = config['feature_dim']
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.ff_dim = config['ff_dim']
        self.dropout = config['dropout']
        self.sequence_length = config['sequence_length']
        self.prediction_horizons = config['prediction_horizons']
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(self.feature_dim, self.embed_dim)
        self.input_norm = nn.LayerNorm(self.embed_dim)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.sequence_length)
        
        # 共享的Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 任务1：稳定性检测头（每个时间尺度一个）
        self.stability_heads = nn.ModuleDict()
        for horizon in self.prediction_horizons:
            self.stability_heads[f'{horizon}min'] = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.embed_dim // 2, 2)  # stable=0, unstable=1
            )
        
        # 任务2：方向判断头（每个时间尺度一个）
        self.direction_heads = nn.ModuleDict()
        for horizon in self.prediction_horizons:
            self.direction_heads[f'{horizon}min'] = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.embed_dim // 2, 2)  # up=0, down=1
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, sequence_length, feature_dim]
            lengths: 序列实际长度 [batch_size] (可选)
            
        Returns:
            两个任务的预测结果字典
        """
        batch_size, seq_len, feature_dim = x.size()
        
        # 输入嵌入和归一化
        x = self.input_embedding(x)  # [batch, seq_len, embed_dim]
        x = self.input_norm(x)
        
        # 位置编码
        x = x.transpose(0, 1)  # [seq_len, batch, embed_dim]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, embed_dim]
        
        # Transformer层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 全局池化：取最后一个时间步的输出
        pooled_output = x[:, -1, :]  # [batch, embed_dim]
        
        # 多任务预测
        predictions = {}
        
        # 任务1：稳定性检测
        for horizon in self.prediction_horizons:
            stability_logits = self.stability_heads[f'{horizon}min'](pooled_output)
            predictions[f'stability_{horizon}min'] = stability_logits
        
        # 任务2：方向判断
        for horizon in self.prediction_horizons:
            direction_logits = self.direction_heads[f'{horizon}min'](pooled_output)
            predictions[f'direction_{horizon}min'] = direction_logits
        
        return predictions


class TwoTaskLoss(nn.Module):
    """两任务损失函数"""
    
    def __init__(self, task_weights: Dict[str, float], 
                 stability_weight: float = 0.4,
                 direction_weight: float = 0.6):
        super().__init__()
        self.task_weights = task_weights
        self.stability_weight = stability_weight  # 稳定性任务权重
        self.direction_weight = direction_weight  # 方向任务权重
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算两任务损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        # 任务1：稳定性检测损失
        stability_loss = 0.0
        stability_count = 0
        
        for task_name, pred_logits in predictions.items():
            if task_name.startswith('stability_'):
                horizon_key = task_name.replace('stability_', '')
                if task_name in targets:
                    target_labels = targets[task_name]
                    task_loss = F.cross_entropy(pred_logits, target_labels)
                    
                    stability_loss += task_loss * self.task_weights.get(horizon_key, 1.0)
                    stability_count += 1
                    losses[f'loss_{task_name}'] = task_loss
        
        if stability_count > 0:
            stability_loss /= stability_count
            total_loss += self.stability_weight * stability_loss
            losses['stability_loss'] = stability_loss
        
        # 任务2：方向判断损失（只在有效样本上计算）
        direction_loss = 0.0
        direction_count = 0
        
        for task_name, pred_logits in predictions.items():
            if task_name.startswith('direction_'):
                horizon_key = task_name.replace('direction_', '')
                if task_name in targets:
                    target_labels = targets[task_name]
                    
                    # 过滤掉无效标签（-1）
                    valid_mask = target_labels != -1
                    if valid_mask.sum() > 0:
                        valid_pred = pred_logits[valid_mask]
                        valid_target = target_labels[valid_mask]
                        
                        task_loss = F.cross_entropy(valid_pred, valid_target)
                        direction_loss += task_loss * self.task_weights.get(horizon_key, 1.0)
                        direction_count += 1
                        losses[f'loss_{task_name}'] = task_loss
        
        if direction_count > 0:
            direction_loss /= direction_count
            total_loss += self.direction_weight * direction_loss
            losses['direction_loss'] = direction_loss
        
        losses['total_loss'] = total_loss
        return losses

if __name__ == "__main__":
    test_transformer_model() 