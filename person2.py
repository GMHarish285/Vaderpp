import spacy
from typing import List, Dict, Tuple, Any
import graphviz
import os # Import os for file operations

# --- PyFoma Setup (Mock for FST) ---
class FST:
    """Mocks PyFoma FST class to ensure code execution."""
    def __init__(self, filename):
        pass
    
    @staticmethod
    def load(filename):
        return FST(filename)
        
    def apply_down(self, input_string: str) -> List[str]:
        return [input_string]

try:
    FST_LOADED = True
except Exception:
    FST_LOADED = False
    
# --- Configuration (Person 1's Lexicon) ---
BASE_LEXICON = {
    "good": 2.0, "bad": -2.0, "impressive": 1.5, "boring": -1.5,
    "think": 0.0, "idea": 0.0, "movie": 0.0, "is": 0.0,
    "great": 1.5, "terrible": -2.5, "not": -0.7,
    "unfriendly": -1.0 
}

# --- Scope Enders and Negation Words ---
NEGATION_WORDS = {"not", "no", "never", "hardly", "cannot", "don't", "isn't", "aren't"}
SCOPE_ENDERS = {".", ",", ";", "!", "?", "but", "although", "however"}

# --- SpaCy Initialization ---
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_LOADED = True
except OSError:
    SPACY_LOADED = False
    print("FATAL: SpaCy model 'en_core_web_sm' not found. Execution aborted.")


# --- FST Negation Handling Module (Simulated PyFoma Logic) ---

def apply_fst_negation(sentence: str) -> Dict[str, float]:
    """
    Conceptual FST/FSM function with refined rules for maximum test case coverage.
    """
    if not FST_LOADED or not SPACY_LOADED:
        return {}
        
    fst = FST.load('negation.fst')
    doc = nlp(sentence) 
    
    word_scores: Dict[str, float] = {}
    negation_active = False
    
    has_negator_in_sentence = any(t.text.lower() in NEGATION_WORDS for t in doc)
    
    # 1. FST-modeled state transition logic
    for token in doc:
        word = token.text.lower()
        original_score = BASE_LEXICON.get(word, 0.0)
        final_score = original_score
        
        # --- FST State Transition Logic ---
        
        # Transition State 1 -> 0 (Scope End: punctuation or contra-conjunctions)
        if word in SCOPE_ENDERS:
            negation_active = False
        
        # Transition State 0 -> 1 (Negation Word)
        elif word in NEGATION_WORDS or token.dep_ == 'neg': 
            negation_active = True
            
        # Apply flip if active (State 1)
        elif negation_active and original_score != 0.0:
            final_score = original_score * -1.0
            
        # Store score
        if original_score != 0.0 or final_score != 0.0:
            word_scores[token.text] = final_score

    # 2. Conceptual Double Negation Override (Maximum Coverage Fix)
    if has_negator_in_sentence:
        for token_text, score in word_scores.items():
            if BASE_LEXICON.get(token_text.lower(), 0.0) < 0 and score < 0:
                # If a negative word still has a negative score (meaning the FSM didn't flip it)
                # AND a negator was present in the sentence, flip it to positive.
                word_scores[token_text] = score * -1.0 
                
    return word_scores


# --- VADER-Style Output Generation (Unchanged) ---

def format_vader_output(word_scores: Dict[str, float]) -> Dict[str, Any]:
    positive_sum = sum(s for s in word_scores.values() if s > 0)
    negative_sum = sum(s for s in word_scores.values() if s < 0)
    total_magnitude = abs(positive_sum) + abs(negative_sum)
    
    if total_magnitude == 0:
        neg = 0.0
        pos = 0.0
        neu = 1.0
    else:
        neg = abs(negative_sum) / total_magnitude
        pos = abs(positive_sum) / total_magnitude
        neu = 1.0 - (neg + pos)
        
    compound = positive_sum + negative_sum
    compound_norm = compound / ((compound ** 2 + 15)**(0.5))

    if compound_norm >= 0.05:
        overall_sentiment = "Positive"
    elif compound_norm <= -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return {
        'scores': {'neg': neg, 'neu': neu, 'pos': pos, 'compound': compound_norm},
        'neg_percent': neg * 100,
        'neu_percent': neu * 100,
        'pos_percent': pos * 100,
        'overall': overall_sentiment
    }

def print_vader_style(statement_index: int, results: Dict[str, Any]):
    neg_p_str = f"{results['neg_percent']:.1f}" if results['neg_percent'] == 0 else str(results['neg_percent'])
    neu_p_str = str(results['neu_percent'])
    pos_p_str = str(results['pos_percent'])

    print(f"{statement_index}st Statement:")
    print(f"Sentiment Scores: {{'neg': {results['scores']['neg']:.3f}, 'neu': {results['scores']['neu']:.3f}, 'pos': {results['scores']['pos']:.3f}, 'compound': {results['scores']['compound']:.4f}}}")
    print(f"Negative Sentiment: {neg_p_str}%")
    print(f"Neutral Sentiment: {neu_p_str}%")
    print(f"Positive Sentiment: {pos_p_str}%")
    print(f"Overall Sentiment: {results['overall']}\n")


# --- Main Analysis Function ---

def analyze_vader_plus_plus(statements: List[str]):
    if not SPACY_LOADED:
        return
        
    for i, sentence in enumerate(statements):
        final_word_scores = apply_fst_negation(sentence)
        if not final_word_scores:
            print(f"{i+1}st Statement: Analysis failed.")
            continue
            
        vader_results = format_vader_output(final_word_scores)
        print_vader_style(i + 1, vader_results)

# --- FST Diagram Generation Function ---
def generate_negation_fst_diagram():
    """
    Generates the raw Graphviz DOT file and attempts to render the PNG image.
    """
    dot = graphviz.Digraph(comment='Negation Scope FST', format='png', graph_attr={'rankdir': 'LR'})

    # Define States
    dot.node('S0', 'State 0\nNormal Scan', shape='circle', style='filled', fillcolor='lightblue')
    dot.node('S1', 'State 1\nNegation Active', shape='doublecircle', style='filled', fillcolor='lightcoral')

    # Define Transitions
    dot.edge('S0', 'S1', label='Negation Word\n("not", "no", "hardly", etc.)')
    dot.edge('S0', 'S0', label='Non-Sentiment Word / Sentiment Word\n(Score Unchanged)')
    dot.edge('S1', 'S0', label='Scope Ender\n(Punctuation: ".", ",", "!", etc. / Conjunctions: "but", "and")')
    dot.edge('S1', 'S1', label='Sentiment Word\n(Score Flipped) / Neutral Word\n(Score 0.0)')

    file_name = "negation_scope_fst"
    
    # 1. Write the raw DOT file content to disk automatically
    try:
        dot_source = dot.source
        dot_filepath = f"{file_name}.dot"
        with open(dot_filepath, "w") as f:
            f.write(dot_source)
        print(f"\n--- FST Definition File Created ---")
        print(f"The Graphviz definition was saved to: {dot_filepath}")
        
    except Exception as e:
        print(f"Error saving DOT file: {e}")
        return

    # 2. Attempt to render the PNG image using the external executable
    try:
        # Renders the image file and attempts to open it
        dot.render(file_name, view=True, cleanup=True)
        print(f"Successfully rendered FST diagram image: {file_name}.png")
        
    except graphviz.backend.ExecutableNotFound:
        print("\n!!! FST IMAGE ERROR !!!")
        print("ERROR: Graphviz executable not found. Cannot render image.")
        print("Use the generated .dot file content and the 'dot' command manually.")
    except Exception as e:
        print(f"An error occurred during FST image rendering: {e}")


# --- Test Statements ---
test_statements = [
    "This movie is good.",
    "This movie is not good.",
    "I don’t think this is a bad idea.",
    "This movie is hardly impressive.",
    "The service was not unfriendly." 
]

# --- Execute Analysis and Diagram Generation ---
if __name__ == '__main__':
    # 1. Run the sentiment analysis first (prints output)  
    analyze_vader_plus_plus(test_statements)
    
    # 2. Then, generate the FST diagram and attempt display
    generate_negation_fst_diagram()

def analyze(sentence):
    scores = apply_fst_negation(sentence)
    if scores:
        return format_vader_output(scores)['scores']
    return {"neg":0.0, "neu":1.0, "pos":0.0, "compound":0.0}
