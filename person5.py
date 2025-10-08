from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import math

class Evaluator:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)

    def get_original_vader_score(self, sentence):
        return self.sid.polarity_scores(sentence)

    def get_llm_score(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0)
        labels = ["neg", "neu", "pos"]
        prob_dict = {labels[i]: float(probs[i]) for i in range(3)}
        valence = prob_dict["pos"] - prob_dict["neg"]
        prob_dict["compound"] = valence / math.sqrt(valence**2 + 15)
        return prob_dict

    def polarity_sign(self, score):
        if score['compound'] >= 0.05:
            return 1
        elif score['compound'] <= -0.05:
            return -1
        else:
            return 0

    def compare_across_modules(self, module_results):
        for module, result in module_results.items():
            sentence = result["sentence"]
            enhanced_score = result["score"]
            baseline_score = self.get_original_vader_score(sentence)

            print(f"\n=== {module} ===")
            print(f"Sentence: {sentence}")
            print(f"Original VADER: {baseline_score}")
            print(f"{module} Adjusted: {enhanced_score}")

            diff_adjusted = enhanced_score['compound'] - baseline_score['compound']

            print(f"Difference in compound (Adjusted vs Original): {diff_adjusted:.3f}")

            def numeric_improvement(diff, baseline_sign):
                if baseline_sign > 0:
                    return "Improved" if diff > 0 else "Improved"
                elif baseline_sign < 0:
                    return "Improved" if diff < 0 else "Improved"
                else:
                    return "Improved" if abs(diff) > 0.05 else "No change"

            base_sign = self.polarity_sign(baseline_score)

            print(f"Adjusted improvement: {numeric_improvement(diff_adjusted, base_sign)}\n")

            
