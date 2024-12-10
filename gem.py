import requests
import time

def gen_ans(query, api_key, max_retries=5):
    # API URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCwSumxEudRZNhzkKGcKA1GdAvXLm_5ZfQ"
    
    # Correct API payload
    data = {"contents":[{"parts":[{"text":query}]}]}


    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            # Send POST request
            response = requests.post(url, json=data)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            
            # Extract generated content safely
            generated_text = response_json.get('candidates', [{}])[0].get('content', '')
            if generated_text:
                return generated_text
            else:
                raise ValueError("Invalid response format: 'content' missing")
        
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:  # Too Many Requests
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error occurred: {http_err}")
                break  # Exit loop for non-retryable errors
        
        except Exception as e:
            print(f"Error making request: {e}")
            break  # Exit loop for non-retryable errors
    
    # Return failure message after retries
    return "Failed to generate response after multiple attempts."

# Example usage
api_key = "AIzaSyCwSumxEudRZNhzkKGcKA1GdAvXLm_5ZfQ"
query = "Explain how photosynthesis works."
response = gen_ans(query, api_key)
print(response)
