from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "textattack/bert-base-uncased-MNLI"#"textattack/bert-base-uncased-MNLI"  # Alternative: "ynie/bert-base-snli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def verify_fact_bert(claim, evidence):
    inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)

    # Assumption: index 1 = entailment (true), index 0 = contradiction (false) – may vary by model
    labels = ['Contradiction', 'Neutral', 'Entailment']
    predicted_label = labels[probs.argmax().item()]
    
    return predicted_label, probs.max().item()
