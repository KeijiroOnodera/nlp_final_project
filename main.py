"""
Main Execution Script
SST-2感情分類タスクにおけるBERTとfastTextの比較実験のメインスクリプト
"""

import os
import argparse
import logging
import json
from datetime import datetime
import numpy as np
import torch

from data_loader import SST2DataLoader
from bert_classifier import BERTClassifier
from fasttext_classifier import FastTextClassifier
from evaluate import ModelEvaluator, print_evaluation_results
from analysis import DetailedAnalyzer
from error_analysis import ErrorAnalyzer

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """再現性のためのシード設定"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results_to_json(results, filepath):
    """結果をJSONファイルに保存"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {filepath}")


def main(args):
    """メイン実行関数"""
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    
    # シード設定
    set_seed(args.seed)
    
    logger.info("="*80)
    logger.info("SST-2 Sentiment Analysis: BERT vs fastText")
    logger.info("="*80)
    
    # ===============================
    # 1. データセットのロード
    # ===============================
    logger.info("\n[Step 1] Loading SST-2 Dataset...")
    data_loader = SST2DataLoader()
    dataset = data_loader.load_data()
    
    # データセット統計の保存
    stats = data_loader.get_statistics()
    print("\nDataset Statistics:")
    print(stats)
    stats.to_csv(os.path.join(args.output_dir, 'results', 'dataset_stats.csv'), 
                 index=False)
    
    # データセットのサブセット化（必要に応じて）
    if args.use_subset:
        logger.info(f"Using subset: {args.subset_ratio*100}% of training data")
        train_data = data_loader.get_subset('train', args.subset_ratio)
    else:
        train_data = dataset['train']
    
    val_data = dataset['validation']
    
    # ===============================
    # 2. モデルの学習
    # ===============================
    
    # --- BERT Classifier ---
    if not args.skip_bert:
        logger.info("\n[Step 2a] Training BERT Classifier...")
        
        bert_model_path = os.path.join(args.output_dir, 'models', 'bert_model')
        
        if args.load_bert and os.path.exists(bert_model_path):
            logger.info(f"Loading pre-trained BERT model from {bert_model_path}")
            bert_classifier = BERTClassifier()
            bert_classifier.load_model(bert_model_path)
        else:
            bert_classifier = BERTClassifier(max_length=args.bert_max_length)
            
            bert_history = bert_classifier.train(
                train_data,
                val_data,
                epochs=args.bert_epochs,
                batch_size=args.bert_batch_size,
                learning_rate=args.bert_lr,
                save_path=bert_model_path
            )
            
            logger.info(f"BERT Training History: {bert_history}")
    
    # --- fastText Classifier ---
    if not args.skip_fasttext:
        logger.info("\n[Step 2b] Training fastText Classifier...")
        
        fasttext_model_path = os.path.join(args.output_dir, 'models', 'fasttext_model')
        
        if args.load_fasttext and os.path.exists(fasttext_model_path + '.bin'):
            logger.info(f"Loading pre-trained fastText model from {fasttext_model_path}")
            fasttext_classifier = FastTextClassifier()
            fasttext_classifier.load_model(fasttext_model_path)
        else:
            fasttext_classifier = FastTextClassifier()
            
            fasttext_history = fasttext_classifier.train(
                train_data,
                val_data,
                lr=args.fasttext_lr,
                epoch=args.fasttext_epochs,
                wordNgrams=args.fasttext_wordngrams,
                dim=args.fasttext_dim,
                save_path=fasttext_model_path
            )
            
            logger.info(f"fastText Training History: {fasttext_history}")
    
    # ===============================
    # 3. 評価
    # ===============================
    logger.info("\n[Step 3] Evaluating Models on Validation Set...")
    
    evaluator = ModelEvaluator()
    predictions = {}
    probabilities = {}
    
    # BERT評価
    if not args.skip_bert:
        logger.info("Evaluating BERT...")
        bert_pred, bert_proba = bert_classifier.predict(val_data, 
                                                        batch_size=args.bert_batch_size)
        predictions['BERT'] = bert_pred
        probabilities['BERT'] = bert_proba
        
        bert_metrics = evaluator.compute_metrics(
            np.array(val_data['label']), 
            bert_pred, 
            'BERT'
        )
        print_evaluation_results(bert_metrics, 'BERT')
        
        # 混同行列の保存
        evaluator.plot_confusion_matrix(
            np.array(val_data['label']),
            bert_pred,
            'BERT',
            save_path=os.path.join(args.output_dir, 'figures', 'bert_confusion_matrix.png')
        )
    
    # fastText評価
    if not args.skip_fasttext:
        logger.info("Evaluating fastText...")
        fasttext_pred, fasttext_proba = fasttext_classifier.predict(val_data)
        predictions['fastText'] = fasttext_pred
        probabilities['fastText'] = fasttext_proba
        
        fasttext_metrics = evaluator.compute_metrics(
            np.array(val_data['label']), 
            fasttext_pred, 
            'fastText'
        )
        print_evaluation_results(fasttext_metrics, 'fastText')
        
        # 混同行列の保存
        evaluator.plot_confusion_matrix(
            np.array(val_data['label']),
            fasttext_pred,
            'fastText',
            save_path=os.path.join(args.output_dir, 'figures', 'fasttext_confusion_matrix.png')
        )
    
    # モデル比較
    if len(predictions) >= 2:
        comparison_df = evaluator.compare_models()
        print("\n=== Model Comparison ===")
        print(comparison_df)
        comparison_df.to_csv(
            os.path.join(args.output_dir, 'results', 'model_comparison.csv'),
            index=False
        )
        
        # 比較グラフの保存
        evaluator.plot_model_comparison(
            save_path=os.path.join(args.output_dir, 'figures', 'model_comparison.png')
        )
    
    # ===============================
    # 4. 詳細分析
    # ===============================
    logger.info("\n[Step 4] Detailed Analysis...")
    
    analyzer = DetailedAnalyzer()
    
    # 文長別分析
    if len(predictions) >= 1:
        logger.info("Analyzing by sentence length...")
        for model_name, y_pred in predictions.items():
            length_results = analyzer.analyze_by_sentence_length(val_data, y_pred)
            print(f"\n{model_name} - Performance by Sentence Length:")
            print(length_results)
            length_results.to_csv(
                os.path.join(args.output_dir, 'results', 
                           f'{model_name.lower()}_length_analysis.csv'),
                index=False
            )
        
        # 複数モデルの比較
        if len(predictions) >= 2:
            analyzer.compare_models_by_length(
                val_data,
                predictions,
                save_path=os.path.join(args.output_dir, 'figures', 
                                      'length_comparison.png')
            )
    
    # 否定表現分析
    if len(predictions) >= 1:
        logger.info("Analyzing negation sentences...")
        for model_name, y_pred in predictions.items():
            negation_results = analyzer.analyze_negation_sentences(val_data, y_pred)
            print(f"\n{model_name} - Performance on Negation Sentences:")
            for key, value in negation_results.items():
                print(f"  {key}: {value}")
            
            # 否定表現の例を取得
            negation_examples = analyzer.get_negation_examples(val_data, y_pred, 
                                                              num_examples=20)
            negation_examples.to_csv(
                os.path.join(args.output_dir, 'results',
                           f'{model_name.lower()}_negation_examples.csv'),
                index=False
            )
        
        # 複数モデルの比較
        if len(predictions) >= 2:
            analyzer.compare_models_by_negation(
                val_data,
                predictions,
                save_path=os.path.join(args.output_dir, 'figures',
                                      'negation_comparison.png')
            )
    
    # ===============================
    # 5. 誤り分析
    # ===============================
    logger.info("\n[Step 5] Error Analysis...")
    
    error_analyzer = ErrorAnalyzer()
    
    # エラーサマリーの生成
    error_summary = error_analyzer.generate_error_summary(
        val_data,
        predictions,
        probabilities
    )
    print("\n" + error_summary)
    
    with open(os.path.join(args.output_dir, 'results', 'error_summary.txt'), 
              'w', encoding='utf-8') as f:
        f.write(error_summary)
    
    # モデル間の不一致例
    if len(predictions) >= 2:
        logger.info("Finding disagreement examples between models...")
        disagreements = error_analyzer.find_disagreement_examples(
            val_data,
            predictions,
            num_examples=20
        )
        
        for pattern, data in disagreements.items():
            print(f"\n{pattern}: {data['count']} cases")
            print(data['examples'].head(10))
            
            data['examples'].to_csv(
                os.path.join(args.output_dir, 'results', 
                           f'disagreement_{pattern}.csv'),
                index=False
            )
    
    # 各モデルの高信頼度エラー
    for model_name in predictions.keys():
        if model_name in probabilities:
            high_conf_errors = error_analyzer.find_high_confidence_errors(
                val_data,
                predictions[model_name],
                probabilities[model_name],
                threshold=0.9,
                num_examples=10
            )
            
            if not high_conf_errors.empty:
                print(f"\n{model_name} - High Confidence Errors:")
                print(high_conf_errors)
                
                high_conf_errors.to_csv(
                    os.path.join(args.output_dir, 'results',
                               f'{model_name.lower()}_high_conf_errors.csv'),
                    index=False
                )
    
    # ===============================
    # 6. 学習データ量による性能変化（オプション）
    # ===============================
    if args.analyze_data_size:
        logger.info("\n[Step 6] Analyzing Performance vs Training Data Size...")
        
        data_ratios = [0.1, 0.25, 0.5, 1.0]
        learning_curve_results = {'BERT': [], 'fastText': []}
        
        for ratio in data_ratios:
            logger.info(f"Training with {ratio*100}% of data...")
            
            train_subset = data_loader.get_subset('train', ratio)
            
            # BERT
            if not args.skip_bert:
                bert_temp = BERTClassifier(max_length=args.bert_max_length)
                bert_temp.train(
                    train_subset,
                    val_data,
                    epochs=2,  # 短縮
                    batch_size=args.bert_batch_size,
                    learning_rate=args.bert_lr
                )
                bert_pred_temp, _ = bert_temp.predict(val_data, 
                                                     batch_size=args.bert_batch_size)
                accuracy = (np.array(val_data['label']) == bert_pred_temp).mean()
                learning_curve_results['BERT'].append(accuracy)
            
            # fastText
            if not args.skip_fasttext:
                fasttext_temp = FastTextClassifier()
                fasttext_temp.train(
                    train_subset,
                    val_data,
                    lr=args.fasttext_lr,
                    epoch=args.fasttext_epochs,
                    wordNgrams=args.fasttext_wordngrams
                )
                fasttext_pred_temp, _ = fasttext_temp.predict(val_data)
                accuracy = (np.array(val_data['label']) == fasttext_pred_temp).mean()
                learning_curve_results['fastText'].append(accuracy)
        
        # 学習曲線のプロット
        analyzer.plot_learning_curve(
            data_ratios,
            learning_curve_results,
            save_path=os.path.join(args.output_dir, 'figures', 'learning_curve.png')
        )
    
    logger.info("\n" + "="*80)
    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SST-2 Sentiment Analysis: BERT vs fastText'
    )
    
    # 基本設定
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--use_subset', action='store_true',
                       help='Use subset of training data')
    parser.add_argument('--subset_ratio', type=float, default=0.1,
                       help='Ratio of training data to use (0.0-1.0)')
    
    # BERT設定
    parser.add_argument('--skip_bert', action='store_true',
                       help='Skip BERT training and evaluation')
    parser.add_argument('--load_bert', action='store_true',
                       help='Load pre-trained BERT model')
    parser.add_argument('--bert_epochs', type=int, default=3,
                       help='Number of epochs for BERT')
    parser.add_argument('--bert_batch_size', type=int, default=16,
                       help='Batch size for BERT')
    parser.add_argument('--bert_lr', type=float, default=2e-5,
                       help='Learning rate for BERT')
    parser.add_argument('--bert_max_length', type=int, default=128,
                       help='Max sequence length for BERT')
    
    # fastText設定
    parser.add_argument('--skip_fasttext', action='store_true',
                       help='Skip fastText training and evaluation')
    parser.add_argument('--load_fasttext', action='store_true',
                       help='Load pre-trained fastText model')
    parser.add_argument('--fasttext_epochs', type=int, default=25,
                       help='Number of epochs for fastText')
    parser.add_argument('--fasttext_lr', type=float, default=0.1,
                       help='Learning rate for fastText')
    parser.add_argument('--fasttext_wordngrams', type=int, default=2,
                       help='Word n-grams for fastText')
    parser.add_argument('--fasttext_dim', type=int, default=100,
                       help='Dimension of word vectors for fastText')
    
    # 分析設定
    parser.add_argument('--analyze_data_size', action='store_true',
                       help='Analyze performance vs training data size')
    
    args = parser.parse_args()
    
    main(args)
