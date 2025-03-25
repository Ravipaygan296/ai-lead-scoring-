import openai

OPENAI_API_KEY = "your_api_key"

def summarize_lead(lead_data):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize lead insights for a sales team."},
            {"role": "user", "content": f"Lead Data: {lead_data}"}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Example Usage
lead_info = {"name": "John Doe", "score": 0.85}
print(summarize_lead(lead_info))
