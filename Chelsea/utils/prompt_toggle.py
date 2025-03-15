import os
import re
import yaml

from typing import Any, Dict, List
from langchain.prompts import PromptTemplate

cwd: str = os.getcwd()
propmt_file_path: str = os.path.join(cwd, "utils/prompts.yaml")

# Load prompts from yaml
def load_prompts():
    try:
        with open(propmt_file_path, "r") as f:
            return yaml.safe_load(f)['prompts']
    except Exception as e:
        print(f"Reading prompts file has failed {e}")

# Preprocess text and keywords
def __preprocess_text(text: str) -> List[Any]:
    return re.findall(r'\b\w+\b', text.lower())

def __preprocess_keywords(keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
    preprocessed_keywords = {}
    for category, kw_list in keywords.items():
        preprocessed_keywords[category] = set(kw.lower() for kw in kw_list)
    return preprocessed_keywords

# Check for keywords in input text
def __check_for_keywords(text: str, keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
    preprocessed_keywords = __preprocess_keywords(keywords)
    matched_keywords = {category: [] for category in keywords}
    words = __preprocess_text(text)
    
    for word in words:
        for category, kw_set in preprocessed_keywords.items():
            if word in kw_set:
                matched_keywords[category].append(word)
    
    matched_keywords = {category: list(set(matches)) for category, matches in matched_keywords.items() if matches}
    
    return matched_keywords

# Select the most appropriate prompt based on matched keywords
def select_prompt(input_text: str, prompts: Any, keywords: Dict[str, List[str]]) -> str:
    matched_keywords = __check_for_keywords(input_text, keywords)
    matched_categories = list(matched_keywords.keys())
    
    # Default to the highest rated common prompt if no specific category is matched
    selected_prompt = max((p for p in prompts if 'common' in p['purpose'] or 'загальні' in p['purpose']), key=lambda p: p['rate'], default=None)
    
    for category in matched_categories:
        category_prompts = [p for p in prompts if category in p['purpose']]
        if category_prompts:
            selected_prompt = max(category_prompts, key=lambda p: p['rate'], default=selected_prompt)

    prompt_template = PromptTemplate(template=selected_prompt['prompt_template'], input_variables=['entity'])
    prompt = prompt_template.format(entity=input_text)
    return prompt