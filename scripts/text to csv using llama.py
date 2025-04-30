import re
import json
import subprocess
import os

# Function to split articles based on "--- Page X ---"
def split_articles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split by "--- Page X ---" to get each article individually
    articles = re.split(r"--- Page \d+ ---", content)
    return [article.strip() for article in articles if article.strip()]

# Function to create prompt for Ollama model
def create_prompt_for_article(article_text):
    prompt_text = f"""
Please analyze the following article and extract the specified fields. Output ONLY the result in valid JSON format with the exact keys as follows: "title", "date", "source", and "content".

Instructions:
- Format each field strictly as JSON.
- Avoid extra commentary or explanations; output only JSON with no other text.
- Ensure output format exactly matches:
{{
    "title": "Extracted title here",
    "date": "Extracted date here",
    "source": "Extracted source here",
    "content": "Extracted content here"
}}

Here is the article text:
{article_text}

Please output strictly as JSON.
"""
    return prompt_text

# Function to run Ollama prompt and get the response
def run_ollama_prompt(model_name, prompt_text):
    try:
        command = ['ollama', 'run', model_name]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr = process.communicate(input=prompt_text)

        if process.returncode != 0:
            print(f"Error: {stderr}")
            return None

        return stdout.strip()
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to enforce JSON format and validate the output
def enforce_json_format(output):
    # Attempt to find JSON using regex to capture from the first '{' to the last '}'
    json_match = re.search(r'{.*}', output, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Error: Could not decode as JSON.")
            return None
    else:
        print("Error: JSON not found in output.")
    return None

# Function to process all articles from a given text file using the model
def process_articles_with_model(file_path, model_name):
    articles = split_articles(file_path)
    structured_articles = []

    for i, article_text in enumerate(articles):
        print(f"Processing article {i+1}/{len(articles)}")
        prompt_text = create_prompt_for_article(article_text)
        output = run_ollama_prompt(model_name, prompt_text)

        # Enforce JSON formatting and validation
        structured_data = enforce_json_format(output)
        if structured_data:
            structured_articles.append(structured_data)
        else:
            print(f"Warning: Could not parse output for article {i+1} as JSON.")

    return structured_articles

# Function to save the processed articles as JSONL
def save_as_jsonl(structured_articles, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for article in structured_articles:
            json.dump(article, outfile)
            outfile.write('\n')

# Function to process all text files in the directory
def process_all_text_files(input_directory, model_name, output_file):
    structured_articles = []
    
    # Process each text file in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):  # Only process .txt files
            file_path = os.path.join(input_directory, filename)
            print(f"Processing file: {filename}")
            structured_data = process_articles_with_model(file_path, model_name)
            structured_articles.extend(structured_data)

    # Save the structured articles to JSONL
    save_as_jsonl(structured_articles, output_file)
    print(f"\nStructured articles saved to {output_file}")

# Example usage
input_directory = r"C:\Users\balaj\code_files\Documents\Brahmanda\context_aware_risk_methodology\event_causal_prediction_system\scripts\pdf_to_text_data_sample"  # Replace with your directory path
model_name = "llama3.1"  # Change to your Ollama model name
output_file = r"C:\Users\balaj\code_files\Documents\Brahmanda\context_aware_risk_methodology\event_causal_prediction_system\data\structured_articles.jsonl"  # Output path

# Process all text files in the specified directory and save to JSONL
process_all_text_files(input_directory, model_name, output_file)
