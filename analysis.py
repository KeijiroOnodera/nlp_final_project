"""
Detailed Analysis Module
文長別、否定表現、学習データ量別の詳細分析を行うモジュール
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetailedAnalyzer:
    """詳細分析を行うクラス"""
    
    def __init__(self):
        """分析器の初期化"""
        self.negation_words = [
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 
            'neither', 'none', 'hardly', 'scarcely', 'barely',
            "n't", "cannot", "can't", "won't", "wouldn't", "shouldn't",
            "couldn't", "doesn't", "didn't", "don't", "isn't", "aren't",
            "wasn't", "weren't", "hasn't", "haven't", "hadn't"
        ]
    
    def analyze_by_sentence_length(self, dataset, y_pred, 
                                   length_bins=None) -> pd.DataFrame:
        """
        文長別の性能分析
        
        Args:
            dataset: Hugging Face Dataset
            y_pred: 予測ラベル
            length_bins: 文長のビン区切り（デフォルト: [0, 10, 20, 30, 100]）
        
        Returns:
            pd.DataFrame: 文長別の評価結果
        """
        if length_bins is None:
            length_bins = [0, 10, 20, 30, 100]
        
        # 文長を計算
        sentence_lengths = [len(sent.split()) for sent in dataset['sentence']]
        y_true = np.array(dataset['label'])
        
        # ビンに分類
        length_categories = pd.cut(sentence_lengths, bins=length_bins, 
                                  labels=[f"{length_bins[i]}-{length_bins[i+1]}" 
                                         for i in range(len(length_bins)-1)])
        
        # 各カテゴリで評価
        results = []
        for category in length_categories.categories:
            mask = (length_categories == category)
            if mask.sum() == 0:
                continue
            
            y_true_subset = y_true[mask]
            y_pred_subset = y_pred[mask]
            
            accuracy = (y_true_subset == y_pred_subset).mean()
            
            results.append({
                'Length Range': category,
                'Sample Count': mask.sum(),
                'Accuracy': f"{accuracy:.4f}",
                'Correct': (y_true_subset == y_pred_subset).sum(),
                'Total': len(y_true_subset)
            })
        
        return pd.DataFrame(results)
    
    def analyze_negation_sentences(self, dataset, y_pred) -> Dict:
        """
        否定表現を含む文の分析
        
        Args:
            dataset: Hugging Face Dataset
            y_pred: 予測ラベル
        
        Returns:
            dict: 否定表現の有無別の評価結果
        """
        y_true = np.array(dataset['label'])
        sentences = dataset['sentence']
        
        # 否定表現を含むかチェック
        has_negation = []
        for sent in sentences:
            sent_lower = sent.lower()
            contains_neg = any(neg_word in sent_lower.split() or 
                             neg_word in sent_lower 
                             for neg_word in self.negation_words)
            has_negation.append(contains_neg)
        
        has_negation = np.array(has_negation)
        
        # 否定表現あり/なしで評価
        results = {}
        
        # 否定表現あり
        neg_mask = has_negation
        if neg_mask.sum() > 0:
            y_true_neg = y_true[neg_mask]
            y_pred_neg = y_pred[neg_mask]
            results['with_negation'] = {
                'count': neg_mask.sum(),
                'accuracy': (y_true_neg == y_pred_neg).mean(),
                'correct': (y_true_neg == y_pred_neg).sum(),
                'total': len(y_true_neg)
            }
        
        # 否定表現なし
        no_neg_mask = ~has_negation
        if no_neg_mask.sum() > 0:
            y_true_no_neg = y_true[no_neg_mask]
            y_pred_no_neg = y_pred[no_neg_mask]
            results['without_negation'] = {
                'count': no_neg_mask.sum(),
                'accuracy': (y_true_no_neg == y_pred_no_neg).mean(),
                'correct': (y_true_no_neg == y_pred_no_neg).sum(),
                'total': len(y_true_no_neg)
            }
        
        return results
    
    def get_negation_examples(self, dataset, y_pred, 
                             num_examples=10) -> pd.DataFrame:
        """
        否定表現を含む文の例を取得
        
        Args:
            dataset: Hugging Face Dataset
            y_pred: 予測ラベル
            num_examples: 取得する例の数
        
        Returns:
            pd.DataFrame: 否定表現を含む文の例
        """
        y_true = np.array(dataset['label'])
        sentences = dataset['sentence']
        
        # 否定表現を含む文を抽出
        examples = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            contains_neg = any(neg_word in sent_lower.split() or 
                             neg_word in sent_lower 
                             for neg_word in self.negation_words)
            
            if contains_neg and len(examples) < num_examples:
                examples.append({
                    'Sentence': sent,
                    'True Label': 'Positive' if y_true[i] == 1 else 'Negative',
                    'Predicted': 'Positive' if y_pred[i] == 1 else 'Negative',
                    'Correct': '✓' if y_true[i] == y_pred[i] else '✗'
                })
        
        return pd.DataFrame(examples)
    
    def compare_models_by_length(self, dataset, predictions_dict, 
                                length_bins=None, save_path=None):
        """
        複数モデルの文長別性能を比較
        
        Args:
            dataset: Hugging Face Dataset
            predictions_dict: {モデル名: 予測ラベル} の辞書
            length_bins: 文長のビン区切り
            save_path: グラフ保存パス
        """
        if length_bins is None:
            length_bins = [0, 10, 20, 30, 100]
        
        sentence_lengths = [len(sent.split()) for sent in dataset['sentence']]
        y_true = np.array(dataset['label'])
        
        length_categories = pd.cut(sentence_lengths, bins=length_bins,
                                  labels=[f"{length_bins[i]}-{length_bins[i+1]}" 
                                         for i in range(len(length_bins)-1)])
        
        # 各モデル、各カテゴリの精度を計算
        results = {model: [] for model in predictions_dict.keys()}
        categories = []
        
        for category in length_categories.categories:
            mask = (length_categories == category)
            if mask.sum() == 0:
                continue
            
            categories.append(str(category))
            y_true_subset = y_true[mask]
            
            for model_name, y_pred in predictions_dict.items():
                y_pred_subset = y_pred[mask]
                accuracy = (y_true_subset == y_pred_subset).mean()
                results[model_name].append(accuracy)
        
        # プロット
        plt.figure(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.35
        
        for i, (model_name, accuracies) in enumerate(results.items()):
            offset = width * (i - 0.5)
            plt.bar(x + offset, accuracies, width, label=model_name, alpha=0.8)
        
        plt.xlabel('Sentence Length Range (words)')
        plt.ylabel('Accuracy')
        plt.title('Model Performance by Sentence Length')
        plt.xticks(x, categories)
        plt.legend()
        plt.ylim([0, 1.0])
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Length comparison plot saved to {save_path}")
        
        plt.close()
    
    def compare_models_by_negation(self, dataset, predictions_dict, 
                                  save_path=None):
        """
        複数モデルの否定表現に対する性能を比較
        
        Args:
            dataset: Hugging Face Dataset
            predictions_dict: {モデル名: 予測ラベル} の辞書
            save_path: グラフ保存パス
        """
        y_true = np.array(dataset['label'])
        sentences = dataset['sentence']
        
        # 否定表現を含むかチェック
        has_negation = []
        for sent in sentences:
            sent_lower = sent.lower()
            contains_neg = any(neg_word in sent_lower.split() or 
                             neg_word in sent_lower 
                             for neg_word in self.negation_words)
            has_negation.append(contains_neg)
        
        has_negation = np.array(has_negation)
        
        # 各モデルの精度を計算
        results = {'With Negation': [], 'Without Negation': []}
        model_names = list(predictions_dict.keys())
        
        for model_name, y_pred in predictions_dict.items():
            # 否定表現あり
            neg_mask = has_negation
            if neg_mask.sum() > 0:
                acc_with_neg = (y_true[neg_mask] == y_pred[neg_mask]).mean()
                results['With Negation'].append(acc_with_neg)
            
            # 否定表現なし
            no_neg_mask = ~has_negation
            if no_neg_mask.sum() > 0:
                acc_without_neg = (y_true[no_neg_mask] == y_pred[no_neg_mask]).mean()
                results['Without Negation'].append(acc_without_neg)
        
        # プロット
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, results['With Negation'], width, 
              label='With Negation', alpha=0.8)
        ax.bar(x + width/2, results['Without Negation'], width, 
              label='Without Negation', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance on Sentences with/without Negation')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Negation comparison plot saved to {save_path}")
        
        plt.close()
    
    def plot_learning_curve(self, data_ratios, results_dict, 
                          metric='accuracy', save_path=None):
        """
        学習曲線（データ量 vs 性能）をプロット
        
        Args:
            data_ratios: データ量の割合リスト（例: [0.1, 0.25, 0.5, 1.0]）
            results_dict: {モデル名: [各データ量での精度]} の辞書
            metric: 評価指標名
            save_path: グラフ保存パス
        """
        plt.figure(figsize=(10, 6))
        
        for model_name, scores in results_dict.items():
            plt.plot([r*100 for r in data_ratios], scores, 
                    marker='o', label=model_name, linewidth=2)
        
        plt.xlabel('Training Data Size (%)')
        plt.ylabel(f'{metric.capitalize()}')
        plt.title(f'Learning Curve: {metric.capitalize()} vs Training Data Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.0])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curve saved to {save_path}")
        
        plt.close()


if __name__ == "__main__":
    # テスト実行
    from data_loader import SST2DataLoader
    
    loader = SST2DataLoader()
    data = loader.load_data()
    
    analyzer = DetailedAnalyzer()
    
    # ダミーの予測で文長別分析をテスト
    val_data = data['validation']
    dummy_pred = np.random.randint(0, 2, len(val_data))
    
    length_results = analyzer.analyze_by_sentence_length(val_data, dummy_pred)
    print("\n=== Analysis by Sentence Length ===")
    print(length_results)
    
    negation_results = analyzer.analyze_negation_sentences(val_data, dummy_pred)
    print("\n=== Analysis of Negation Sentences ===")
    print(negation_results)
