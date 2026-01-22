"""
Data Loader Module
SST-2データセットをHugging Face Datasetsから読み込むモジュール
"""

from datasets import load_dataset
import pandas as pd
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SST2DataLoader:
    """SST-2データセットのロードと前処理を担当するクラス"""
    
    def __init__(self):
        """データセットの初期化"""
        self.dataset = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
    def load_data(self) -> Dict:
        """
        Hugging Face DatasetsからSST-2データセットをロード
        
        Returns:
            Dict: train, validation, testの各分割を含む辞書
        """
        logger.info("Loading SST-2 dataset from Hugging Face Datasets...")
        
        # GLUEベンチマークからSST-2をロード
        self.dataset = load_dataset("glue", "sst2")
        
        self.train_data = self.dataset["train"]
        self.validation_data = self.dataset["validation"]
        self.test_data = self.dataset["test"]
        
        logger.info(f"Train samples: {len(self.train_data)}")
        logger.info(f"Validation samples: {len(self.validation_data)}")
        logger.info(f"Test samples: {len(self.test_data)}")
        
        return {
            "train": self.train_data,
            "validation": self.validation_data,
            "test": self.test_data
        }
    
    def get_subset(self, split: str, ratio: float = 1.0) -> Dict:
        """
        データセットの一部を取得（データ量を減らした実験用）
        
        Args:
            split: "train", "validation", "test"のいずれか
            ratio: 使用するデータの割合（0.0~1.0）
        
        Returns:
            Dict: 指定された割合のデータセット
        """
        if self.dataset is None:
            self.load_data()
        
        data = self.dataset[split]
        
        if ratio < 1.0:
            num_samples = int(len(data) * ratio)
            data = data.select(range(num_samples))
            logger.info(f"Using {num_samples} samples ({ratio*100:.0f}%) from {split} split")
        
        return data
    
    def get_statistics(self) -> pd.DataFrame:
        """
        データセットの統計情報を取得
        
        Returns:
            pd.DataFrame: 統計情報を含むDataFrame
        """
        if self.dataset is None:
            self.load_data()
        
        stats = []
        
        for split_name in ["train", "validation", "test"]:
            split_data = self.dataset[split_name]
            
            # ラベル分布を計算（testはラベルが-1の場合がある）
            if split_name != "test" or split_data["label"][0] != -1:
                positive_count = sum(1 for label in split_data["label"] if label == 1)
                negative_count = sum(1 for label in split_data["label"] if label == 0)
                
                stats.append({
                    "Split": split_name,
                    "Total Samples": len(split_data),
                    "Positive": positive_count,
                    "Negative": negative_count,
                    "Positive Ratio": f"{positive_count/len(split_data)*100:.2f}%"
                })
            else:
                stats.append({
                    "Split": split_name,
                    "Total Samples": len(split_data),
                    "Positive": "N/A (no labels)",
                    "Negative": "N/A (no labels)",
                    "Positive Ratio": "N/A"
                })
        
        return pd.DataFrame(stats)
    
    def get_sentence_lengths(self, split: str = "validation") -> list:
        """
        文長の分布を取得
        
        Args:
            split: 分析対象の分割
        
        Returns:
            list: 各文の単語数のリスト
        """
        if self.dataset is None:
            self.load_data()
        
        data = self.dataset[split]
        sentence_lengths = [len(sentence.split()) for sentence in data["sentence"]]
        
        return sentence_lengths


def prepare_fasttext_format(dataset, output_file: str):
    """
    fastText用のフォーマットでデータを保存
    
    Args:
        dataset: Hugging Face Dataset
        output_file: 出力ファイルパス
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            # fastTextの形式: __label__<label> <text>
            label = example['label']
            text = example['sentence'].replace('\n', ' ')
            f.write(f"__label__{label} {text}\n")
    
    logger.info(f"Saved fastText format data to {output_file}")


if __name__ == "__main__":
    # テスト実行
    loader = SST2DataLoader()
    data = loader.load_data()
    
    print("\n=== Dataset Statistics ===")
    print(loader.get_statistics())
    
    print("\n=== Sample Data ===")
    print(f"Train example: {data['train'][0]}")
    print(f"Validation example: {data['validation'][0]}")
