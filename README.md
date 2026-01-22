# SST-2 Sentiment Analysis: BERT vs fastText

SST-2 (Stanford Sentiment Treebank)データセットを用いた感情分析タスクにおいて、**BERT**と**fastText**の2つの異なるアプローチを比較する自然言語処理プロジェクトです。

## 📊 プロジェクト概要

このプロジェクトでは、以下を実施します：

- **2つのモデルの学習と評価**
  - **BERT** (bert-base-uncased): 事前学習済みTransformerモデル
  - **fastText**: 高速な単語埋め込みベースのテキスト分類器

- **詳細な性能比較分析**
  - Accuracy、Precision、Recall、F1スコア
  - 混同行列
  - 学習曲線

- **エラー分析**
  - 文章長による性能差
  - 否定表現への対応力
  - モデル間の判断の相違

## 🗂️ プロジェクト構造

```
nlp_final_project/
├── main.py                    # メイン実行スクリプト
├── data_loader.py             # SST-2データセットのロードと前処理
├── bert_classifier.py         # BERT分類器の実装
├── fasttext_classifier.py     # fastText分類器の実装
├── evaluate.py                # モデル評価機能
├── analysis.py                # 詳細分析と可視化
├── error_analysis.py          # エラー分析
├── requirements.txt           # 依存パッケージ
└── outputs/                   # 実行結果の出力先
    ├── models/                # 学習済みモデル
    ├── figures/               # 可視化グラフ
    └── results/               # 評価結果CSV
```

## 🚀 セットアップ

### 1. 環境構築

**Python 3.8以上**が必要です。仮想環境の作成を推奨します：

```bash
# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

**主な依存パッケージ:**
- `torch>=2.0.0` - PyTorch深層学習フレームワーク
- `transformers>=4.30.0` - Hugging Face Transformersライブラリ
- `datasets>=2.14.0` - データセットライブラリ
- `fasttext>=0.9.2` - fastTextライブラリ
- `scikit-learn>=1.3.0` - 機械学習ユーティリティ
- `matplotlib`, `seaborn` - データ可視化

### 3. GPU環境（推奨）

BERTの学習にはGPUの使用を強く推奨します。GPU環境の確認：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 📖 使用方法

### 基本的な実行

```bash
python main.py
```

### よく使うオプション

#### データのサブセットで高速実行（テスト用）

```bash
python main.py --use_subset --subset_ratio 0.1
```

#### 特定のモデルのみを実行

```bash
# BERTのみ
python main.py --skip_fasttext

# fastTextのみ
python main.py --skip_bert
```

#### 学習済みモデルの読み込み

```bash
python main.py --load_bert --load_fasttext
```

#### ハイパーパラメータの調整

```bash
python main.py \
  --bert_epochs 5 \
  --bert_batch_size 32 \
  --bert_learning_rate 2e-5 \
  --fasttext_epochs 30
```

#### データサイズによる性能変化の分析

```bash
python main.py --analyze_data_size
```

### 全オプション一覧

```
使用可能なオプション:
  --output_dir DIR          出力ディレクトリ (デフォルト: ./outputs)
  --seed SEED               乱数シード (デフォルト: 42)
  --use_subset              データセットのサブセットを使用
  --subset_ratio RATIO      サブセットの割合 (デフォルト: 0.1)
  
  BERT設定:
  --skip_bert               BERTの学習をスキップ
  --load_bert               保存済みBERTモデルをロード
  --bert_epochs N           エポック数 (デフォルト: 3)
  --bert_batch_size N       バッチサイズ (デフォルト: 16)
  --bert_learning_rate LR   学習率 (デフォルト: 2e-5)
  --bert_max_length N       最大トークン長 (デフォルト: 128)
  
  fastText設定:
  --skip_fasttext           fastTextの学習をスキップ
  --load_fasttext           保存済みfastTextモデルをロード
  --fasttext_epochs N       エポック数 (デフォルト: 25)
  --fasttext_lr LR          学習率 (デフォルト: 0.1)
  --fasttext_dim N          埋め込み次元数 (デフォルト: 100)
  
  分析:
  --analyze_data_size       データサイズによる性能変化を分析
```

## 📈 出力結果

実行後、`outputs/`ディレクトリに以下が保存されます：

### 1. モデルファイル (`outputs/models/`)
- `bert_model/` - 学習済みBERTモデル
- `fasttext_model.bin` - 学習済みfastTextモデル

### 2. 可視化グラフ (`outputs/figures/`)
- `model_comparison.png` - モデル性能比較（Accuracy, F1など）
- `bert_confusion_matrix.png` - BERT混同行列
- `fasttext_confusion_matrix.png` - fastText混同行列
- `length_comparison.png` - 文章長別の性能比較
- `negation_comparison.png` - 否定表現への対応力比較

### 3. 評価結果 (`outputs/results/`)
- `model_comparison.csv` - モデル性能比較表
- `dataset_stats.csv` - データセット統計
- `error_summary.txt` - エラー分析サマリー
- `*_high_conf_errors.csv` - 高信頼度エラー事例
- `*_length_analysis.csv` - 文章長別分析結果
- `*_negation_examples.csv` - 否定表現の分析結果
- `disagreement_*.csv` - モデル間の判断の相違事例

## 📊 実験結果の例

実行例（サブセット10%使用）：

| Model    | Accuracy | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| BERT     | 0.89     | 0.90      | 0.88   | 0.89     |
| fastText | 0.82     | 0.81      | 0.83   | 0.82     |

**主な発見:**
- BERTは文脈理解に優れ、複雑な表現や否定文に強い
- fastTextは学習・推論が高速で、単純な文に対しては十分な性能
- 長文になるほどBERTの優位性が顕著

## 🔍 技術詳細

### BERTアプローチ
- **モデル**: `bert-base-uncased`（Hugging Face）
- **アーキテクチャ**: 12層Transformer、768次元
- **トークナイザー**: WordPieceトークナイザー
- **最適化**: AdamW、学習率2e-5
- **Fine-tuning**: 分類層のみ追加して全体をファインチューニング

### fastTextアプローチ
- **アーキテクチャ**: 単語埋め込みの平均 + 線形分類器
- **特徴**: サブワード情報の活用（character n-grams）
- **最適化**: SGD、学習率0.1
- **利点**: 高速な学習・推論、OOV（未知語）への対応力

### データセット
- **SST-2**: Stanford Sentiment Treebank v2
- **タスク**: 二値感情分類（Positive/Negative）
- **データ量**: 
  - Training: 67,349サンプル
  - Validation: 872サンプル
  - Test: 1,821サンプル（ラベルなし）

## 🛠️ トラブルシューティング

### Python 2.7を使用しているエラー

```bash
# Python 3を明示的に使用
python3 -m pip install -r requirements.txt
python3 main.py
```

### importlib_metadataのエラー（Python 3.8）

```bash
pip install --upgrade 'importlib-metadata>=6.0.0'
```

### GPU/CUDAのエラー

CPUのみで実行する場合、BERTの学習は時間がかかります：

```bash
# CPUで実行（時間がかかる）
CUDA_VISIBLE_DEVICES="" python main.py --use_subset --subset_ratio 0.1
```

### メモリ不足エラー

バッチサイズを小さくしてください：

```bash
python main.py --bert_batch_size 8
```

## 📝 引用

このプロジェクトで使用したデータセットとモデル：

```bibtex
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew and Potts, Christopher},
  booktitle={Proceedings of EMNLP},
  year={2013}
}

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@article{joulin2016fasttext,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

## 📄 ライセンス

このプロジェクトは学術・教育目的で作成されています。

## 👤 作成者

- GitHub: [@KeijiroOnodera](https://github.com/KeijiroOnodera)
- プロジェクト: [nlp_final_project](https://github.com/KeijiroOnodera/nlp_final_project)

## 🙏 謝辞

- Hugging Face Transformersライブラリ
- Facebook Research fastTextライブラリ
- Stanford NLP Group（SST-2データセット）
