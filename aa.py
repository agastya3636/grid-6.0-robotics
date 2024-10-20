import requests

def gen_ans(query):
    # API URL and API key (replace with your actual key)
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyDQz5dgbtryYd3MG_sDxfdFBPtL9JeBIPU"
    
    # Data to be sent (input for the language model)
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": query
                    }
                ]
            }
        ]
    }

    try:
        # Send the POST request
        response = requests.post(url, json=data)
        response.raise_for_status()  # Check for HTTP errors

        # Extract and print the generated response
        generated_text = response.json()['candidates'][0]['content']
        print("Generated Answer:", generated_text)

    except requests.exceptions.RequestException as e:
        print("Error making request:", e)

# Call the function
gen_ans(query="Can you give the roadmap of DSA within 200 words?")
