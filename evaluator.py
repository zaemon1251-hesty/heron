from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import evaluate


class Evaler:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        # self.meteor_metric = evaluate.load("meteor")
        try:
            from nlgeval import NLGEval

            self.caption_scorer = NLGEval(no_glove=True, no_skipthoughts=True)
        except Exception:
            pass

        try:
            self.metrics = {
                "blue": evaluate.load("bleu"),
                "meteor": evaluate.load("meteor"),
                # "rouge": evaluate.load("rouge"),
                # "precision": evaluate.load("precision"),
                # "recall": evaluate.load("recall"),
            }
        except Exception:
            import traceback

            print(traceback.format_exc())
            print("Failed to load metrics [Evaluate]")
            pass

    def compute_metrics_a(self, eval_pred):
        """Computes metrics for evaluation
        Args:
            eval_pred (tuple):
            eval_pred[0] にはモデルの予測が入る。
            eval_pred[1] には_get_item_inferenceメソッドの戻り値 (inputs, imgs, answer) が入る。具体的には:
            eval_pred[1][0] は inputs
            eval_pred[1][2] は answer

        Returns:
            _type_: _description_
        """
        # モデルの予測を取得
        predictions = eval_pred[0]

        # 真のキャプションを取得
        all_labels = [eval_pred[1][2]]  # `answer`から

        # 予測をデコードして実際のキャプションを取得する
        all_outputs = [
            self.tokenizer.decode(pred, skip_special_tokens=True)
            for pred in predictions
        ]

        # スコアを計算する
        scores = self.caption_scorer.compute_metrics(
            ref_list=[all_labels], hyp_list=all_outputs
        )

        return scores

    def compute_metrics_b(self, eval_pred):
        print(eval_pred)

        predictions, labels = eval_pred
        print(f"predictions: {predictions}\n labels: {labels}")

        decoded_predictions = [
            self.tokenizer.decode(pred, skip_special_tokens=True)
            for pred in predictions
        ]
        decoded_labels = [
            self.tokenizer.decode(label, skip_special_tokens=True) for label in labels
        ]

        print(decoded_labels, decoded_predictions)

        # BLEU scores (maybe not correct)
        smoothing = SmoothingFunction().method1
        bleu_1 = sum(
            [
                sentence_bleu(
                    [ref], pred, weights=(1, 0, 0, 0), smoothing_function=smoothing
                )
                for ref, pred in zip(decoded_labels, decoded_predictions)
            ]
        ) / len(decoded_labels)
        bleu_2 = sum(
            [
                sentence_bleu(
                    [ref], pred, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing
                )
                for ref, pred in zip(decoded_labels, decoded_predictions)
            ]
        ) / len(decoded_labels)
        bleu_3 = sum(
            [
                sentence_bleu(
                    [ref],
                    pred,
                    weights=(0.33, 0.33, 0.33, 0),
                    smoothing_function=smoothing,
                )
                for ref, pred in zip(decoded_labels, decoded_predictions)
            ]
        ) / len(decoded_labels)
        bleu_4 = sum(
            [
                sentence_bleu(
                    [ref],
                    pred,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothing,
                )
                for ref, pred in zip(decoded_labels, decoded_predictions)
            ]
        ) / len(decoded_labels)

        # METEOR
        # meteor = self.meteor_metric.compute(
        #     predictions=decoded_labels, references=decoded_predictions
        # )

        # ROUGE_L
        rouge = Rouge()
        scores = rouge.get_scores(decoded_predictions, decoded_labels, avg=True)
        rouge_l = scores["rouge-l"]["f"]

        # TODO: CIDEr, Recall, and Precision calculation goes here...

        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
            "meteor": None,
            "rouge_l": rouge_l,
            # ... [other metrics]
        }

    def compute_metrics_c(self, eval_pred):
        predictions, labels = eval_pred

        decoded_predictions = [
            self.tokenizer.decode(pred, skip_special_tokens=True)
            for pred in predictions
        ]
        decoded_labels = [
            self.tokenizer.decode(label, skip_special_tokens=True) for label in labels
        ]
        result = {}
        for metric in self.metrics:
            result[metric] = self.metrics[metric].compute(
                predictions=decoded_labels, references=decoded_predictions
            )
        return result
