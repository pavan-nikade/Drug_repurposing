import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b-instruct"

def explain_path(drug, gene, pathway, disease):
    # Your improved, clean presentation prompt
    prompt = f'''
You are explaining a biomedical graph path to a computer science audience.

Drug: {drug}
Gene: {gene}
Pathway: {pathway}
Disease: {disease}

Rules:
- Explain in exactly 4-6 short sentences.
- Do not mention entropy, topology scores, convergence, or any metric.
- Do not invent additional genes, pathways, or biological facts.
- Use simple engineering-friendly language.
- Use cautious words such as may, could, suggests, or plausible.
- End with the exact phrase: "This is supportive graph-based evidence, not proof of treatment."
'''

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        # OPTIMIZED: Combined your safe options while giving token headroom
        "options": {
            "temperature": 0.1,    # Forces strict factual logic
            "num_predict": 250     # Increased to prevent truncating mid-sentence
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        json_data = response.json()
        return json_data["message"]["content"].strip()
    except Exception as e:
        return f"System Execution Fault: {e}"
