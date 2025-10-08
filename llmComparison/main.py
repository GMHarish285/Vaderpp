import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import pandas as pd

# --- Your Unmodified RoBERTa Class ---
class RoBERTaSentiment:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        neg, neu, pos = probs[0].item(), probs[1].item(), probs[2].item()
        compound = pos - neg
        return {"neg": neg, "neu": neu, "pos": pos, "compound": compound}

# --- Refined Gemma Class ---
class GemmaSentiment:
    def __init__(self, model_name="google/gemma-2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.labels = ["Positive", "Negative", "Neutral"]
        self.labels_lower = [label.lower() for label in self.labels]

    def analyze(self, text):
        messages = [{"role": "user", "content": f"Analyze the sentiment of the following sentence. Respond with only one word: Positive, Negative, or Neutral.\n\nSentence: '{text}'"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**input_ids, max_new_tokens=5, temperature=0.1)
        response_ids = outputs[0][input_ids.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        predicted_label = "Unknown"
        first_word = response_text.split()[0].lower() if response_text else ""
        if first_word in self.labels_lower:
            predicted_label = self.labels[self.labels_lower.index(first_word)]
        else:
            for label in self.labels:
                if label.lower() in response_text.lower():
                    predicted_label = label
                    break
        return predicted_label, response_text

# --- Main Execution Block for Comparison ---
if __name__ == '__main__':
    print("Loading models... (This may take a while, especially for Gemma)")
    roberta_pipeline = RoBERTaSentiment()
    gemma_pipeline = GemmaSentiment()
    print("All models loaded successfully!")

    test_dataset = [
        "The customer service was excellent and very helpful.",
        "I had a frustrating and terrible experience with their support team.",
        "Oh, great. Another software update that broke more than it fixed.",
        "The system requires a software update to version 3.1.",
        "The stock soared after the bullish earnings report.",
    ]
    
    results = []
    print("\n--- Running Comparative Analysis ---")
    for i, sentence in enumerate(test_dataset):
        print(f"\nProcessing sentence {i+1}/{len(test_dataset)}: '{sentence}'")
        
        # RoBERTa Analysis
        roberta_scores = roberta_pipeline.analyze(sentence)
        # Determine label by finding the key with the max value
        roberta_label = max(roberta_scores, key=lambda k: roberta_scores[k] if k != 'compound' else -1)
        
        # Gemma Analysis
        gemma_label, _ = gemma_pipeline.analyze(sentence)
        
        results.append({
            "Sentence": sentence,
            "RoBERTa Label": roberta_label.capitalize(),
            "RoBERTa Scores (Probabilities)": f"Pos: {roberta_scores['pos']:.4f}, Neu: {roberta_scores['neu']:.4f}, Neg: {roberta_scores['neg']:.4f}",
            "Gemma Label": gemma_label
        })

    # Display results in a clean table using pandas
    df = pd.DataFrame(results)
    print("\n\n--- Comparative Results ---")
    print(df.to_string())