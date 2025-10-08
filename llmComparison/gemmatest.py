import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GemmaSentiment:
    def __init__(self, model_name="google/gemma-2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.labels = ["Positive", "Negative", "Neutral"]
        self.labels_lower = [label.lower() for label in self.labels]

    def analyze(self, text):
        # --- CHANGE 1: Use the official chat template for better prompting ---
        # This formats the prompt in the exact way the model was trained.
        messages = [
            {
                "role": "user",
                "content": (
                    f"Analyze the sentiment of the following sentence. "
                    f"Respond with only one word: Positive, Negative, or Neutral.\n\n"
                    f"Sentence: '{text}'"
                )
            }
        ]
        
        # The 'add_generation_prompt=True' adds the special tokens to signal the model to start generating.
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=5, # Slightly increased for safety
                temperature=0.1
            )
        
        response_ids = outputs[0][input_ids.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # --- CHANGE 2: Implement stricter response parsing ---
        # Check if the first word of the response matches one of our labels.
        predicted_label = "Unknown"
        first_word = response_text.split()[0].lower() if response_text else ""
        
        if first_word in self.labels_lower:
            # Capitalize to match the original format (e.g., "positive" -> "Positive")
            predicted_label = self.labels[self.labels_lower.index(first_word)]
        else:
            # Fallback for safety, same as your original logic
             for label in self.labels:
                if label.lower() in response_text.lower():
                    predicted_label = label
                    break

        return predicted_label, response_text

# --- Main execution block (No changes needed here) ---
if __name__ == '__main__':
    print("Loading Gemma model... (This may take a while)")
    gemma_pipeline = GemmaSentiment()
    print("Model loaded successfully!")

    test_dataset = [
        # 1. Simple Positive
        "The customer service was excellent and very helpful.",
        # 2. Simple Negative
        "I had a frustrating and terrible experience with their support team.",
        # 3. Nuanced / Sarcastic
        "Oh, great. Another software update that broke more than it fixed.",
        # 4. Mixed Sentiment
        "The food was amazing, but the long wait time was a huge letdown.",
        # 5. Neutral / Factual
        "The system requires a software update to version 3.1.",
        # 6. Domain-Specific (Finance)
        "The stock soared after the bullish earnings report.",
        # 7. Domain-Specific (Medicine)
        "Adverse side-effects were reported, which is a major complication.",
        # 8. Subtle Positive
        "It's not the worst phone I've ever used.",
        # 9. Subtle Negative
        "The movie was anything but entertaining."
    ]

    print("\n--- Analyzing Sentences ---")
    for i, sentence in enumerate(test_dataset):
        predicted_sentiment, raw_output = gemma_pipeline.analyze(sentence)
        print(f"{i+1}. Sentence: '{sentence}'")
        print(f"   Predicted Sentiment: {predicted_sentiment}\n")