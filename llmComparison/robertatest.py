import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RoBERTaSentiment:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        # The model's config provides the label mapping
        self.labels = self.model.config.id2label

    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        
        # Create a dictionary of labels and their scores
        scores = {self.labels[i]: probs[i].item() for i in range(len(probs))}
        
        # Get the predicted label
        pred_index = torch.argmax(probs).item()
        pred_label = self.labels[pred_index]
        
        # Calculate compound score for convenience
        compound = scores.get('positive', 0.0) - scores.get('negative', 0.0)
        
        return pred_label, scores, compound

# --- Main execution block ---
if __name__ == '__main__':
    # Initialize the sentiment analysis pipeline
    print("Loading RoBERTa model...")
    roberta_pipeline = RoBERTaSentiment()
    print("Model loaded successfully!")

    # A diverse dataset to test the model
    test_dataset = [
        "The customer service was excellent and very helpful.",
        "I had a frustrating and terrible experience with their support team.",
        "Oh, great. Another software update that broke more than it fixed.",
        "The food was amazing, but the long wait time was a huge letdown.",
        "The system requires a software update to version 3.1.",
        "The stock soared after the bullish earnings report.",
        "Adverse side-effects were reported, which is a major complication.",
        "It's not the worst phone I've ever used.",
        "The movie was anything but entertaining."
    ]

    print("\n--- Analyzing Sentences ---")
    for i, sentence in enumerate(test_dataset):
        predicted_label, scores, compound_score = roberta_pipeline.analyze(sentence)
        print(f"{i+1}. Sentence: '{sentence}'")
        print(f"   Predicted Label: {predicted_label}")
        # Formatting the scores for cleaner output
        formatted_scores = ", ".join([f"{label}: {score:.2f}" for label, score in scores.items()])
        print(f"   Scores: {formatted_scores}")
        print(f"   Compound Score: {compound_score:.2f}\n")