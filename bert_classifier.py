"""
BERT Classifier Module
bert-base-uncasedを使用した感情分類器の実装
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SST2Dataset(Dataset):
    """BERT用のデータセットクラス"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier:
    """BERT分類器のラッパークラス"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2, max_length=128):
        """
        Args:
            model_name: 使用するBERTモデル
            num_labels: 分類ラベル数
            max_length: 最大トークン長
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # トークナイザーとモデルの初期化
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
    def prepare_data(self, dataset, batch_size=16):
        """
        データセットをDataLoaderに変換
        
        Args:
            dataset: Hugging Face Dataset
            batch_size: バッチサイズ
        
        Returns:
            DataLoader
        """
        texts = dataset['sentence']
        labels = dataset['label']
        
        sst2_dataset = SST2Dataset(texts, labels, self.tokenizer, self.max_length)
        
        return DataLoader(sst2_dataset, batch_size=batch_size, shuffle=False)
    
    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, 
              learning_rate=2e-5, warmup_steps=500, save_path=None):
        """
        モデルの学習
        
        Args:
            train_dataset: 訓練データ
            val_dataset: 検証データ
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            warmup_steps: ウォームアップステップ数
            save_path: モデル保存パス
        
        Returns:
            dict: 学習履歴
        """
        logger.info("Preparing data loaders...")
        train_loader = self.prepare_data(train_dataset, batch_size)
        val_loader = self.prepare_data(val_dataset, batch_size)
        
        # オプティマイザーとスケジューラーの設定
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch in train_progress:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_progress.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss, val_accuracy = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}")
        
        # モデルの保存
        if save_path:
            self.save_model(save_path)
        
        return history
    
    def evaluate(self, data_loader):
        """
        モデルの評価
        
        Args:
            data_loader: 評価用DataLoader
        
        Returns:
            tuple: (平均損失, 正解率)
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def predict(self, dataset, batch_size=16):
        """
        予測の実行
        
        Args:
            dataset: Hugging Face Dataset
            batch_size: バッチサイズ
        
        Returns:
            tuple: (予測ラベル, 予測確率)
        """
        data_loader = self.prepare_data(dataset, batch_size)
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def save_model(self, save_path):
        """モデルの保存"""
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path):
        """モデルの読み込み"""
        logger.info(f"Loading model from {load_path}")
        self.model = BertForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.model.to(self.device)


if __name__ == "__main__":
    # テスト実行
    from data_loader import SST2DataLoader
    
    loader = SST2DataLoader()
    data = loader.load_data()
    
    # 小さなサブセットでテスト
    train_subset = data['train'].select(range(100))
    val_subset = data['validation'].select(range(50))
    
    classifier = BERTClassifier()
    logger.info("BERT Classifier initialized successfully")
