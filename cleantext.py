import re
import unicodedata

def extract_pure_text(text):
    
    normalized_text = unicodedata.normalize('NFC', text)
    
    
    pure_text = re.sub(r'[^a-zA-Z0-9\s.,!]', '', normalized_text)
    
    
    pure_text = re.sub(r'\s+', ' ', pure_text).strip()
    
    return pure_text

"""# Example usage
text = "I ❤️ Python! It's great for #coding123, and more... 😊💻🚀"
clean_text = extract_pure_text(text)
print(clean_text)
"""