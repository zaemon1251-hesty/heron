import unittest

# `compute_metrics`と必要な依存関係をインポート
from evaluator import Evaler
from transformers import AutoTokenizer


class TestComputeMetrics(unittest.TestCase):
    def setUp(self):
        # 例のためのトークナイザの初期化
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", verbose=True)
        self.tokenizer = tokenizer
        # NLGの評価ツールのインスタンス化
        self.evaler = Evaler(tokenizer)

    def test_basic(self):
        # 基本的な動作テスト

        # 例の入力と予測
        predictions = [self.tokenizer.encode("This is a sample prediction.")]
        labels = [self.tokenizer.encode("This is a sample reference caption.")]

        # compute_metrics関数の呼び出し
        eval_pred = (predictions, labels)
        scores = self.evaler.compute_metrics_c(eval_pred)

        # スコアのデータ型を確認
        self.assertIsInstance(scores, dict)

        # 期待されるメトリック（例: BLEU）のスコアが存在することを確認
        # self.assertIn("bleu_1", scores)

    def test_unexpected_input(self):
        # 異常系テスト: Noneの入力

        with self.assertRaises(Exception):
            self.evaler.compute_metrics_b((None, None))


if __name__ == "__main__":
    unittest.main()
