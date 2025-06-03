"""
交易感知评估指标工具
专门针对金融预测的评估方法，重点关注方向性预测准确性
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算方向准确率（只考虑涨跌，忽略持平）
    
    Args:
        y_true: 真实标签 [0=涨, 1=跌, 2=持平]
        y_pred: 预测标签 [0=涨, 1=跌, 2=持平]
        
    Returns:
        方向准确率 (0-1)
    """
    # 过滤出涨跌样本（非持平）
    non_flat_mask = (y_true != 2) & (y_pred != 2)
    
    if non_flat_mask.sum() == 0:
        return 0.0
    
    y_true_filtered = y_true[non_flat_mask]
    y_pred_filtered = y_pred[non_flat_mask]
    
    return accuracy_score(y_true_filtered, y_pred_filtered)


def calculate_catastrophic_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算灾难性错误率（方向完全相反的预测）
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        错误率统计字典
    """
    total_samples = len(y_true)
    
    # 方向性错误
    up_to_down = ((y_pred == 0) & (y_true == 1)).sum()  # 预测涨实际跌
    down_to_up = ((y_pred == 1) & (y_true == 0)).sum()  # 预测跌实际涨
    
    catastrophic_errors = up_to_down + down_to_up
    catastrophic_rate = catastrophic_errors / total_samples
    
    return {
        'catastrophic_error_rate': catastrophic_rate,
        'up_to_down_errors': up_to_down,
        'down_to_up_errors': down_to_up,
        'total_catastrophic': catastrophic_errors,
        'total_samples': total_samples
    }


def calculate_trading_value_score(y_true: np.ndarray, y_pred: np.ndarray, 
                                error_weights: Dict[str, float]) -> Dict[str, float]:
    """
    计算交易价值评分
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签  
        error_weights: 错误权重配置
        
    Returns:
        评分统计
    """
    total_score = 0.0
    error_counts = {
        'correct': 0,
        'up_to_down': 0,
        'down_to_up': 0,
        'up_to_flat': 0,
        'down_to_flat': 0,
        'flat_to_up': 0,
        'flat_to_down': 0
    }
    
    for i in range(len(y_true)):
        true_label, pred_label = y_true[i], y_pred[i]
        
        if true_label == pred_label:
            # 预测正确
            total_score += error_weights['correct_prediction']
            error_counts['correct'] += 1
        elif true_label == 0 and pred_label == 1:
            # 预测涨实际跌
            total_score += error_weights['up_to_down_error']
            error_counts['up_to_down'] += 1
        elif true_label == 1 and pred_label == 0:
            # 预测跌实际涨
            total_score += error_weights['down_to_up_error']
            error_counts['down_to_up'] += 1
        elif true_label == 0 and pred_label == 2:
            # 预测涨实际平
            total_score += error_weights['up_to_flat_error']
            error_counts['up_to_flat'] += 1
        elif true_label == 1 and pred_label == 2:
            # 预测跌实际平
            total_score += error_weights['down_to_flat_error']
            error_counts['down_to_flat'] += 1
        elif true_label == 2 and pred_label == 0:
            # 预测平实际涨
            total_score += error_weights['flat_to_up_error']
            error_counts['flat_to_up'] += 1
        elif true_label == 2 and pred_label == 1:
            # 预测平实际跌
            total_score += error_weights['flat_to_down_error']
            error_counts['flat_to_down'] += 1
    
    normalized_score = total_score / len(y_true)
    
    return {
        'trading_value_score': normalized_score,
        'total_score': total_score,
        'error_counts': error_counts
    }


def calculate_class_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算各类别的精确率和召回率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        各类别指标
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )
    
    return {
        'up_precision': precision[0],
        'down_precision': precision[1], 
        'flat_precision': precision[2],
        'up_recall': recall[0],
        'down_recall': recall[1],
        'flat_recall': recall[2],
        'up_f1': f1[0],
        'down_f1': f1[1],
        'flat_f1': f1[2],
        'up_support': support[0],
        'down_support': support[1],
        'flat_support': support[2]
    }


def comprehensive_evaluation(y_true: np.ndarray, y_pred: np.ndarray, 
                           error_weights: Dict[str, float]) -> Dict[str, any]:
    """
    综合评估函数
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        error_weights: 错误权重配置
        
    Returns:
        完整评估结果
    """
    results = {}
    
    # 基础指标
    results['overall_accuracy'] = accuracy_score(y_true, y_pred)
    
    # 方向性指标
    results['directional_accuracy'] = calculate_directional_accuracy(y_true, y_pred)
    
    # 灾难性错误
    catastrophic_metrics = calculate_catastrophic_error_rate(y_true, y_pred)
    results.update(catastrophic_metrics)
    
    # 交易价值评分
    trading_metrics = calculate_trading_value_score(y_true, y_pred, error_weights)
    results.update(trading_metrics)
    
    # 类别特定指标
    class_metrics = calculate_class_specific_metrics(y_true, y_pred)
    results.update(class_metrics)
    
    # 混淆矩阵
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return results


def should_continue_training(metrics: Dict[str, float], config) -> Tuple[bool, str]:
    """
    智能早停判断
    
    Args:
        metrics: 当前评估指标
        config: 配置对象
        
    Returns:
        (是否继续训练, 停止原因)
    """
    # 检查灾难性错误率是否过高
    if metrics['catastrophic_error_rate'] > config.max_catastrophic_error_rate:
        return True, f"灾难错误率过高: {metrics['catastrophic_error_rate']:.3f}"
    
    # 检查方向准确率是否太低
    if metrics['directional_accuracy'] < config.min_directional_accuracy:
        return True, f"方向准确率过低: {metrics['directional_accuracy']:.3f}"
    
    # 检查涨跌类召回率
    if metrics.get('up_recall', 0) < 0.1 or metrics.get('down_recall', 0) < 0.1:
        return True, "涨跌类召回率过低，继续训练"
    
    return False, "评估指标满足早停条件"


def print_trading_evaluation_summary(results: Dict[str, any], epoch: int = None):
    """
    打印交易感知评估摘要
    
    Args:
        results: 评估结果
        epoch: 当前轮次
    """
    if epoch is not None:
        print(f"\n📊 Epoch {epoch} - 交易感知评估结果")
    else:
        print("\n📊 交易感知评估结果")
    
    print("="*50)
    
    # 核心指标
    print("🎯 核心交易指标:")
    print(f"  方向准确率: {results['directional_accuracy']:.3f}")
    print(f"  灾难错误率: {results['catastrophic_error_rate']:.3f}")
    print(f"  交易价值评分: {results['trading_value_score']:.3f}")
    
    # 错误分解
    print(f"\n🚨 错误分析:")
    print(f"  预测涨→实际跌: {results['up_to_down_errors']}次 💥")
    print(f"  预测跌→实际涨: {results['down_to_up_errors']}次 💥")
    
    error_counts = results['error_counts']
    print(f"  预测涨→实际平: {error_counts['up_to_flat']}次")
    print(f"  预测跌→实际平: {error_counts['down_to_flat']}次")
    
    # 类别表现
    print(f"\n📈 类别表现:")
    print(f"  上涨类 - 精确率: {results['up_precision']:.3f}, 召回率: {results['up_recall']:.3f}")
    print(f"  下跌类 - 精确率: {results['down_precision']:.3f}, 召回率: {results['down_recall']:.3f}")
    print(f"  持平类 - 精确率: {results['flat_precision']:.3f}, 召回率: {results['flat_recall']:.3f}")
    
    # 样本分布
    print(f"\n📊 样本分布:")
    print(f"  上涨样本: {results['up_support']}个")
    print(f"  下跌样本: {results['down_support']}个") 
    print(f"  持平样本: {results['flat_support']}个")
    print(f"  总体准确率: {results['overall_accuracy']:.3f}")
    
    print("="*50) 