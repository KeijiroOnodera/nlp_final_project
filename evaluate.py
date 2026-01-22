"""
Evaluation Module
モデルの性能評価を行うモジュール
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """モデル評価を行うクラス"""
    
    def __init__(self):
        """評価器の初期化"""
        self.results = {}
    
    def compute_metrics(self, y_true, y_pred, model_name="Model"):
        """
        各種評価指標を計算
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            model_name: モデル名
        
        Returns:
            dict: 評価指標の辞書
        """
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='binary'),
            'Recall': recall_score(y_true, y_pred, average='binary'),
            'F1-Score': f1_score(y_true, y_pred, average='binary')
        }
        
        # 各クラスごとの指標も計算
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics['Precision_Negative'] = precision_per_class[0]
        metrics['Precision_Positive'] = precision_per_class[1]
        metrics['Recall_Negative'] = recall_per_class[0]
        metrics['Recall_Positive'] = recall_per_class[1]
        metrics['F1_Negative'] = f1_per_class[0]
        metrics['F1_Positive'] = f1_per_class[1]
        
        self.results[model_name] = metrics
        
        return metrics
    
    def get_confusion_matrix(self, y_true, y_pred):
        """
        混同行列を取得
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
        
        Returns:
            numpy.ndarray: 混同行列
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", 
                            save_path=None, figsize=(8, 6)):
        """
        混同行列を可視化
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            model_name: モデル名
            save_path: 保存パス
            figsize: 図のサイズ
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def get_classification_report(self, y_true, y_pred, target_names=None):
        """
        詳細な分類レポートを取得
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            target_names: クラス名のリスト
        
        Returns:
            str: 分類レポート
        """
        if target_names is None:
            target_names = ['Negative', 'Positive']
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def compare_models(self, results_dict=None):
        """
        複数モデルの性能を比較
        
        Args:
            results_dict: モデル名をキー、評価指標辞書を値とする辞書
                         Noneの場合は self.results を使用
        
        Returns:
            pd.DataFrame: 比較結果のDataFrame
        """
        if results_dict is None:
            results_dict = self.results
        
        if not results_dict:
            logger.warning("No results to compare")
            return None
        
        # 主要な指標のみを抽出
        comparison_data = []
        for model_name, metrics in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1-Score': f"{metrics['F1-Score']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_model_comparison(self, results_dict=None, save_path=None, 
                            figsize=(12, 6)):
        """
        複数モデルの性能を棒グラフで比較
        
        Args:
            results_dict: モデル名をキー、評価指標辞書を値とする辞書
            save_path: 保存パス
            figsize: 図のサイズ
        """
        if results_dict is None:
            results_dict = self.results
        
        if not results_dict:
            logger.warning("No results to plot")
            return
        
        # データの準備
        models = list(results_dict.keys())
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        data = {metric: [results_dict[model][metric] for model in models] 
                for metric in metrics_names}
        
        # プロット
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, metric in enumerate(metrics_names):
            offset = width * (i - 1.5)
            ax.bar(x + offset, data[metric], width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.close()
    
    def analyze_errors(self, y_true, y_pred, texts, model_name="Model", 
                      num_examples=10):
        """
        誤分類の例を分析
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル
            texts: テキストデータ
            model_name: モデル名
            num_examples: 表示する例の数
        
        Returns:
            pd.DataFrame: 誤分類の例
        """
        # 誤分類のインデックスを取得
        error_indices = np.where(y_true != y_pred)[0]
        
        if len(error_indices) == 0:
            logger.info(f"{model_name}: No errors found!")
            return None
        
        # ランダムにサンプリング
        if len(error_indices) > num_examples:
            sample_indices = np.random.choice(error_indices, num_examples, 
                                            replace=False)
        else:
            sample_indices = error_indices
        
        error_examples = []
        for idx in sample_indices:
            error_examples.append({
                'Text': texts[idx],
                'True Label': 'Positive' if y_true[idx] == 1 else 'Negative',
                'Predicted': 'Positive' if y_pred[idx] == 1 else 'Negative'
            })
        
        df = pd.DataFrame(error_examples)
        
        logger.info(f"{model_name} - Total errors: {len(error_indices)} "
                   f"({len(error_indices)/len(y_true)*100:.2f}%)")
        
        return df
    
    def get_detailed_comparison(self):
        """
        クラスごとの詳細な性能比較を取得
        
        Returns:
            pd.DataFrame: 詳細な比較結果
        """
        if not self.results:
            logger.warning("No results available")
            return None
        
        detailed_data = []
        for model_name, metrics in self.results.items():
            for class_label in ['Negative', 'Positive']:
                detailed_data.append({
                    'Model': model_name,
                    'Class': class_label,
                    'Precision': f"{metrics[f'Precision_{class_label}']:.4f}",
                    'Recall': f"{metrics[f'Recall_{class_label}']:.4f}",
                    'F1-Score': f"{metrics[f'F1_{class_label}']:.4f}"
                })
        
        return pd.DataFrame(detailed_data)


def print_evaluation_results(metrics, model_name="Model"):
    """
    評価結果を見やすく出力
    
    Args:
        metrics: 評価指標の辞書
        model_name: モデル名
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # テスト実行
    np.random.seed(42)
    
    # ダミーデータで評価
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(y_true, y_pred, "Test Model")
    print_evaluation_results(metrics, "Test Model")
    
    print("\nClassification Report:")
    print(evaluator.get_classification_report(y_true, y_pred))
