"""
Error Analysis Module
詳細な誤り分析（定性的分析）を行うモジュール
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """誤り分析を行うクラス"""
    
    def __init__(self):
        """誤り分析器の初期化"""
        pass
    
    def find_disagreement_examples(self, dataset, predictions_dict,
                                  num_examples=20) -> Dict:
        """
        モデル間で予測が異なる例を抽出
        
        Args:
            dataset: Hugging Face Dataset
            predictions_dict: {モデル名: 予測ラベル} の辞書
            num_examples: 抽出する例の数
        
        Returns:
            dict: 各パターンの例を含む辞書
        """
        if len(predictions_dict) != 2:
            logger.warning("This method is designed for comparing 2 models")
            return {}
        
        model_names = list(predictions_dict.keys())
        model1_name, model2_name = model_names[0], model_names[1]
        
        y_true = np.array(dataset['label'])
        y_pred1 = predictions_dict[model1_name]
        y_pred2 = predictions_dict[model2_name]
        
        # パターン1: モデル1は正解、モデル2は不正解
        pattern1_mask = (y_pred1 == y_true) & (y_pred2 != y_true)
        pattern1_indices = np.where(pattern1_mask)[0]
        
        # パターン2: モデル1は不正解、モデル2は正解
        pattern2_mask = (y_pred1 != y_true) & (y_pred2 == y_true)
        pattern2_indices = np.where(pattern2_mask)[0]
        
        # パターン3: 両方とも不正解だが予測が異なる
        pattern3_mask = (y_pred1 != y_true) & (y_pred2 != y_true) & (y_pred1 != y_pred2)
        pattern3_indices = np.where(pattern3_mask)[0]
        
        results = {}
        
        # パターン1の例
        if len(pattern1_indices) > 0:
            sample_size = min(num_examples, len(pattern1_indices))
            sampled_indices = np.random.choice(pattern1_indices, sample_size, 
                                              replace=False)
            
            examples = []
            for idx in sampled_indices:
                examples.append({
                    'Sentence': dataset['sentence'][idx],
                    'True Label': 'Positive' if y_true[idx] == 1 else 'Negative',
                    f'{model1_name}': 'Positive' if y_pred1[idx] == 1 else 'Negative',
                    f'{model2_name}': 'Positive' if y_pred2[idx] == 1 else 'Negative'
                })
            
            results[f'{model1_name}_correct_{model2_name}_wrong'] = {
                'count': len(pattern1_indices),
                'examples': pd.DataFrame(examples)
            }
        
        # パターン2の例
        if len(pattern2_indices) > 0:
            sample_size = min(num_examples, len(pattern2_indices))
            sampled_indices = np.random.choice(pattern2_indices, sample_size, 
                                              replace=False)
            
            examples = []
            for idx in sampled_indices:
                examples.append({
                    'Sentence': dataset['sentence'][idx],
                    'True Label': 'Positive' if y_true[idx] == 1 else 'Negative',
                    f'{model1_name}': 'Positive' if y_pred1[idx] == 1 else 'Negative',
                    f'{model2_name}': 'Positive' if y_pred2[idx] == 1 else 'Negative'
                })
            
            results[f'{model1_name}_wrong_{model2_name}_correct'] = {
                'count': len(pattern2_indices),
                'examples': pd.DataFrame(examples)
            }
        
        # パターン3の例
        if len(pattern3_indices) > 0:
            sample_size = min(num_examples, len(pattern3_indices))
            sampled_indices = np.random.choice(pattern3_indices, sample_size, 
                                              replace=False)
            
            examples = []
            for idx in sampled_indices:
                examples.append({
                    'Sentence': dataset['sentence'][idx],
                    'True Label': 'Positive' if y_true[idx] == 1 else 'Negative',
                    f'{model1_name}': 'Positive' if y_pred1[idx] == 1 else 'Negative',
                    f'{model2_name}': 'Positive' if y_pred2[idx] == 1 else 'Negative'
                })
            
            results['both_wrong_different_predictions'] = {
                'count': len(pattern3_indices),
                'examples': pd.DataFrame(examples)
            }
        
        return results
    
    def analyze_error_patterns(self, dataset, y_pred, model_name="Model") -> Dict:
        """
        誤りのパターンを分析
        
        Args:
            dataset: Hugging Face Dataset
            y_pred: 予測ラベル
            model_name: モデル名
        
        Returns:
            dict: エラーパターンの統計
        """
        y_true = np.array(dataset['label'])
        sentences = dataset['sentence']
        
        # False Positive (実際はNegativeなのにPositiveと予測)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fp_indices = np.where(fp_mask)[0]
        
        # False Negative (実際はPositiveなのにNegativeと予測)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_indices = np.where(fn_mask)[0]
        
        # True Positive
        tp_mask = (y_true == 1) & (y_pred == 1)
        tp_count = tp_mask.sum()
        
        # True Negative
        tn_mask = (y_true == 0) & (y_pred == 0)
        tn_count = tn_mask.sum()
        
        results = {
            'model_name': model_name,
            'true_positive': int(tp_count),
            'true_negative': int(tn_count),
            'false_positive': len(fp_indices),
            'false_negative': len(fn_indices),
            'false_positive_rate': len(fp_indices) / (len(fp_indices) + tn_count) if (len(fp_indices) + tn_count) > 0 else 0,
            'false_negative_rate': len(fn_indices) / (len(fn_indices) + tp_count) if (len(fn_indices) + tp_count) > 0 else 0
        }
        
        return results
    
    def get_error_examples_by_type(self, dataset, y_pred, 
                                  error_type='fp', num_examples=10) -> pd.DataFrame:
        """
        特定のエラータイプの例を取得
        
        Args:
            dataset: Hugging Face Dataset
            y_pred: 予測ラベル
            error_type: 'fp' (False Positive) or 'fn' (False Negative)
            num_examples: 取得する例の数
        
        Returns:
            pd.DataFrame: エラー例
        """
        y_true = np.array(dataset['label'])
        sentences = dataset['sentence']
        
        if error_type.lower() == 'fp':
            # False Positive
            mask = (y_true == 0) & (y_pred == 1)
            error_name = "False Positive (Predicted Positive, Actually Negative)"
        elif error_type.lower() == 'fn':
            # False Negative
            mask = (y_true == 1) & (y_pred == 0)
            error_name = "False Negative (Predicted Negative, Actually Positive)"
        else:
            raise ValueError("error_type must be 'fp' or 'fn'")
        
        error_indices = np.where(mask)[0]
        
        if len(error_indices) == 0:
            logger.info(f"No {error_name} errors found!")
            return pd.DataFrame()
        
        # サンプリング
        sample_size = min(num_examples, len(error_indices))
        sampled_indices = np.random.choice(error_indices, sample_size, replace=False)
        
        examples = []
        for idx in sampled_indices:
            examples.append({
                'Sentence': sentences[idx],
                'True Label': 'Positive' if y_true[idx] == 1 else 'Negative',
                'Predicted': 'Positive' if y_pred[idx] == 1 else 'Negative',
                'Length (words)': len(sentences[idx].split())
            })
        
        return pd.DataFrame(examples)
    
    def analyze_confidence_distribution(self, dataset, y_pred, y_proba) -> Dict:
        """
        予測の信頼度分布を分析
        
        Args:
            dataset: Hugging Face Dataset
            y_pred: 予測ラベル
            y_proba: 予測確率 (num_samples, 2)
        
        Returns:
            dict: 信頼度の統計
        """
        y_true = np.array(dataset['label'])
        
        # 予測クラスの確率を取得
        pred_proba = np.max(y_proba, axis=1)
        
        # 正解/不正解別の信頼度
        correct_mask = (y_true == y_pred)
        incorrect_mask = ~correct_mask
        
        results = {
            'correct_predictions': {
                'count': correct_mask.sum(),
                'mean_confidence': float(pred_proba[correct_mask].mean()),
                'std_confidence': float(pred_proba[correct_mask].std()),
                'min_confidence': float(pred_proba[correct_mask].min()),
                'max_confidence': float(pred_proba[correct_mask].max())
            },
            'incorrect_predictions': {
                'count': incorrect_mask.sum(),
                'mean_confidence': float(pred_proba[incorrect_mask].mean()),
                'std_confidence': float(pred_proba[incorrect_mask].std()),
                'min_confidence': float(pred_proba[incorrect_mask].min()),
                'max_confidence': float(pred_proba[incorrect_mask].max())
            }
        }
        
        return results
    
    def find_high_confidence_errors(self, dataset, y_pred, y_proba, 
                                   threshold=0.9, num_examples=10) -> pd.DataFrame:
        """
        高信頼度で誤った予測の例を抽出
        
        Args:
            dataset: Hugging Face Dataset
            y_pred: 予測ラベル
            y_proba: 予測確率
            threshold: 高信頼度の閾値
            num_examples: 取得する例の数
        
        Returns:
            pd.DataFrame: 高信頼度エラーの例
        """
        y_true = np.array(dataset['label'])
        sentences = dataset['sentence']
        
        # 予測クラスの確率を取得
        pred_proba = np.max(y_proba, axis=1)
        
        # 高信頼度で誤った予測
        high_conf_error_mask = (y_true != y_pred) & (pred_proba >= threshold)
        error_indices = np.where(high_conf_error_mask)[0]
        
        if len(error_indices) == 0:
            logger.info(f"No high-confidence errors found (threshold: {threshold})")
            return pd.DataFrame()
        
        # 信頼度の高い順にソート
        sorted_indices = error_indices[np.argsort(-pred_proba[error_indices])]
        
        # サンプリング
        sample_size = min(num_examples, len(sorted_indices))
        sampled_indices = sorted_indices[:sample_size]
        
        examples = []
        for idx in sampled_indices:
            examples.append({
                'Sentence': sentences[idx],
                'True Label': 'Positive' if y_true[idx] == 1 else 'Negative',
                'Predicted': 'Positive' if y_pred[idx] == 1 else 'Negative',
                'Confidence': f"{pred_proba[idx]:.4f}"
            })
        
        return pd.DataFrame(examples)
    
    def generate_error_summary(self, dataset, predictions_dict, 
                             probabilities_dict=None) -> str:
        """
        誤り分析のサマリーを生成
        
        Args:
            dataset: Hugging Face Dataset
            predictions_dict: {モデル名: 予測ラベル} の辞書
            probabilities_dict: {モデル名: 予測確率} の辞書
        
        Returns:
            str: サマリーテキスト
        """
        summary = "=" * 80 + "\n"
        summary += "ERROR ANALYSIS SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        for model_name, y_pred in predictions_dict.items():
            summary += f"[{model_name}]\n"
            summary += "-" * 80 + "\n"
            
            # エラーパターンの統計
            error_stats = self.analyze_error_patterns(dataset, y_pred, model_name)
            
            summary += f"True Positive:  {error_stats['true_positive']}\n"
            summary += f"True Negative:  {error_stats['true_negative']}\n"
            summary += f"False Positive: {error_stats['false_positive']} "
            summary += f"(FPR: {error_stats['false_positive_rate']:.4f})\n"
            summary += f"False Negative: {error_stats['false_negative']} "
            summary += f"(FNR: {error_stats['false_negative_rate']:.4f})\n"
            
            # 信頼度の分析（確率が提供されている場合）
            if probabilities_dict and model_name in probabilities_dict:
                y_proba = probabilities_dict[model_name]
                conf_stats = self.analyze_confidence_distribution(dataset, y_pred, y_proba)
                
                summary += f"\nConfidence Statistics:\n"
                summary += f"  Correct predictions - Mean: {conf_stats['correct_predictions']['mean_confidence']:.4f}, "
                summary += f"Std: {conf_stats['correct_predictions']['std_confidence']:.4f}\n"
                summary += f"  Incorrect predictions - Mean: {conf_stats['incorrect_predictions']['mean_confidence']:.4f}, "
                summary += f"Std: {conf_stats['incorrect_predictions']['std_confidence']:.4f}\n"
            
            summary += "\n"
        
        return summary


if __name__ == "__main__":
    # テスト実行
    from data_loader import SST2DataLoader
    
    loader = SST2DataLoader()
    data = loader.load_data()
    
    analyzer = ErrorAnalyzer()
    
    # ダミーの予測でテスト
    val_data = data['validation']
    dummy_pred1 = np.random.randint(0, 2, len(val_data))
    dummy_pred2 = np.random.randint(0, 2, len(val_data))
    
    predictions_dict = {
        'Model1': dummy_pred1,
        'Model2': dummy_pred2
    }
    
    # 不一致例の抽出
    disagreements = analyzer.find_disagreement_examples(val_data, predictions_dict, 
                                                       num_examples=5)
    
    print("\n=== Disagreement Examples ===")
    for pattern, data in disagreements.items():
        print(f"\n{pattern}: {data['count']} cases")
        print(data['examples'].head())
