import perrson1, person2, person3, person4
from person5 import Evaluator

if __name__ == "__main__":
    evaluator = Evaluator()

    module_results = {
        "Domain-Specific": {
            "sentence": "The stock is highly undervalued and will likely surge.",
            "score": perrson1.domainExpansion("The laptop performance is excellent and the battery lasts long.")
        },
        "Negation": {
            "sentence": "The movie was not good at all.",
            "score": person2.analyze("The movie was not good at all.")
        },
        "Sarcasm": {
            "sentence": "Oh great, another Monday morning traffic jam. Just what I needed!",
            "score": person3.analyze("Oh great, another Monday morning traffic jam. Just what I needed!")
        },
        "Intensifiers": {
            "sentence": "The food was absolutely fantastic!",
            "score": person4.analyze("The food was absolutely fantastic!")
        }
    }

    evaluator.compare_across_modules(module_results)
    
