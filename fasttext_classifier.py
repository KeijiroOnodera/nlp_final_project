"""
fastText Classifier Module
fastTextを使用した感情分類器の実装
"""

import fasttext
import tempfile
import os
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastTextClassifier:
    """fastText分類器のラッパークラス"""
    
    def __init__(self):
        """fastText分類器の初期化"""
        self.model = None
        self.temp_dir = tempfile.mkdtemp()
        
    def _prepare_fasttext_file(self, dataset, filepath):
        """
        fastText用のフォーマットでデータを保存
        
        Args:
            dataset: Hugging Face Dataset
            filepath: 出力ファイルパス
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in dataset:
                label = example['label']
                # テキストの前処理：改行を削除
                text = example['sentence'].replace('\n', ' ').strip()
                # fastTextの形式: __label__<label> <text>
                f.write(f"__label__{label} {text}\n")
        
        logger.info(f"Prepared fastText format file: {filepath}")
    
    def train(self, train_dataset, val_dataset=None, 
              lr=0.1, epoch=25, wordNgrams=2, dim=100, 
              loss='softmax', save_path=None):
        """
        fastTextモデルの学習
        
        Args:
            train_dataset: 訓練データ (Hugging Face Dataset)
            val_dataset: 検証データ (Hugging Face Dataset, optional)
            lr: 学習率
            epoch: エポック数
            wordNgrams: n-gramの最大サイズ
            dim: 単語ベクトルの次元数
            loss: 損失関数 ('softmax', 'ns', 'hs', 'ova')
            save_path: モデル保存パス
        
        Returns:
            dict: 学習履歴（検証データがあれば精度も含む）
        """
        logger.info("Preparing training data for fastText...")
        
        # 一時ファイルにデータを保存
        train_file = os.path.join(self.temp_dir, 'train.txt')
        self._prepare_fasttext_file(train_dataset, train_file)
        
        logger.info(f"Training fastText model with parameters:")
        logger.info(f"  - Learning rate: {lr}")
        logger.info(f"  - Epochs: {epoch}")
        logger.info(f"  - Word n-grams: {wordNgrams}")
        logger.info(f"  - Dimensions: {dim}")
        logger.info(f"  - Loss: {loss}")
        
        # fastTextモデルの学習
        # verbose=2でトレーニングの進捗を表示
        self.model = fasttext.train_supervised(
            input=train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            dim=dim,
            loss=loss,
            verbose=2
        )
        
        logger.info("Training completed!")
        
        # 訓練データでの性能評価
        train_result = self.model.test(train_file)
        train_accuracy = train_result[1]  # (samples, precision, recall)
        
        history = {
            'train_accuracy': train_accuracy,
        }
        
        # 検証データがあれば評価
        if val_dataset is not None:
            val_file = os.path.join(self.temp_dir, 'val.txt')
            self._prepare_fasttext_file(val_dataset, val_file)
            val_result = self.model.test(val_file)
            val_accuracy = val_result[1]
            history['val_accuracy'] = val_accuracy
            logger.info(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        else:
            logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        
        # モデルの保存
        if save_path:
            self.save_model(save_path)
        
        return history
    
    def predict(self, dataset, k=1):
        """
        予測の実行
        
        Args:
            dataset: Hugging Face Dataset
            k: トップk個の予測を返す
        
        Returns:
            tuple: (予測ラベル, 予測確率)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        predictions = []
        probabilities = []
        
        logger.info("Making predictions with fastText...")
        
        for example in tqdm(dataset, desc="Predicting"):
            text = example['sentence'].replace('\n', ' ').strip()
            # fastTextは複数行の予測も可能だが、ここでは1つずつ
            labels, probs = self.model.predict(text, k=2)  # 2クラス分の確率を取得
            
            # ラベルは '__label__0' の形式なので、数値に変換
            label_num = int(labels[0].replace('__label__', ''))
            predictions.append(label_num)
            
            # 確率ベクトルを作成（クラス0とクラス1の確率）
            # fastTextの出力は予測されたラベルの確率のみなので、2クラスの確率配列を構築
            if len(probs) == 2:
                if label_num == 0:
                    prob_vec = [probs[0], probs[1]]
                else:
                    prob_vec = [probs[1], probs[0]]
            else:
                # 1つのクラスしか返されない場合
                if label_num == 0:
                    prob_vec = [probs[0], 1 - probs[0]]
                else:
                    prob_vec = [1 - probs[0], probs[0]]
            
            probabilities.append(prob_vec)
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(self, dataset):
        """
        データセットでの評価
        
        Args:
            dataset: Hugging Face Dataset
        
        Returns:
            tuple: (サンプル数, 精度, 再現率)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        # 一時ファイルに保存して評価
        eval_file = os.path.join(self.temp_dir, 'eval.txt')
        self._prepare_fasttext_file(dataset, eval_file)
        
        result = self.model.test(eval_file)
        # result: (samples, precision, recall)
        
        return result
    
    def save_model(self, save_path):
        """
        モデルの保存
        
        Args:
            save_path: 保存パス（.binが自動的に付加される）
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        # .binを除いたパスで保存（fastTextが自動的に.binを追加）
        if save_path.endswith('.bin'):
            save_path = save_path[:-4]
        
        self.model.save_model(save_path + '.bin')
        logger.info(f"Model saved to {save_path}.bin")
    
    def load_model(self, load_path):
        """
        モデルの読み込み
        
        Args:
            load_path: 読み込みパス
        """
        if not load_path.endswith('.bin'):
            load_path += '.bin'
        
        self.model = fasttext.load_model(load_path)
        logger.info(f"Model loaded from {load_path}")
    
    def get_word_vector(self, word):
        """
        単語ベクトルの取得
        
        Args:
            word: 単語
        
        Returns:
            numpy.ndarray: 単語ベクトル
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        return self.model.get_word_vector(word)
    
    def get_sentence_vector(self, text):
        """
        文ベクトルの取得
        
        Args:
            text: テキスト
        
        Returns:
            numpy.ndarray: 文ベクトル
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        return self.model.get_sentence_vector(text)
    
    def __del__(self):
        """デストラクタ：一時ディレクトリのクリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    # テスト実行
    from data_loader import SST2DataLoader
    
    loader = SST2DataLoader()
    data = loader.load_data()
    
    # 小さなサブセットでテスト
    train_subset = data['train'].select(range(1000))
    val_subset = data['validation'].select(range(100))
    
    classifier = FastTextClassifier()
    logger.info("fastText Classifier initialized successfully")
    
    # テスト学習
    history = classifier.train(train_subset, val_subset, epoch=5)
    logger.info(f"Training history: {history}")
