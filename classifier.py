import torch
import torch.nn as nn
import numpy as np
import shap

# Gereken torch/shap (DL/XAI) kütüphanelerini ekledik. 



SYMPTOMS = ["fever", "sore_throat", "fatigue", "cough", "headache", "nausea", "rash", "chills"]
DISEASES = ["flu", "cold", "covid", "strep_throat", "migraine"]


# Semptomlar/bulgular ve hastalıklar için listeler oluşturduk. 


X_train = torch.tensor([
    [1, 1, 1, 1, 0, 0, 0, 1],  # flu
    [0, 1, 0, 1, 0, 0, 0, 0],  # cold
    [1, 1, 1, 1, 1, 0, 0, 1],  # covid
    [1, 1, 1, 0, 0, 0, 0, 0],  # strep_throat
    [0, 0, 1, 0, 1, 1, 0, 0],  # migraine
    [1, 0, 1, 1, 1, 0, 0, 1],  # flu
    [0, 1, 0, 1, 0, 0, 0, 0],  # cold
    [1, 1, 1, 1, 0, 1, 0, 1],  # covid
], dtype=torch.float32)

y_train = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])


# Satır ve sütunlar sırasıyla, hasta ve semptomları gösteriyor, y_trainde ise de hastaların hastalıkları gösteriliyor.



class SymptomClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, x):
        return self.net(x)


# Lineer modellerle eğitiyoruz, ortasında da karışlık fonksiyonu ekledik.

model = SymptomClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()

# 200 kere modeli eğitiyoruz, tahmin, hesapla, geri yayılım.

def predict(symptom_text: str):
    symptom_text = symptom_text.lower()
    features = [1 if s in symptom_text else 0 for s in SYMPTOMS]
    tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output).item()
    return DISEASES[pred], features

# Semptom metininden 0/1 listesine çevirim, sonra tahmin

def explain(symptom_text: str):
    symptom_text = symptom_text.lower()
    features = [1 if s in symptom_text else 0 for s in SYMPTOMS]

# Model neden o hastalığı seçti diye açıklama XAI (SHAP)

    def model_predict(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            out = model(x_tensor)
            probs = torch.softmax(out, dim=1).numpy()
        return probs

    background = X_train.numpy()
    explainer = shap.KernelExplainer(model_predict, background)
    input_np = np.array(features).reshape(1, -1)
    shap_values = explainer.shap_values(input_np, nsamples=100)

    predicted_disease, _ = predict(symptom_text)

    shap_vals = np.array(shap_values).mean(axis=0).flatten()
    contributions = dict(zip(SYMPTOMS, shap_vals))
    top_factors = sorted(contributions.items(), key=lambda x: float(abs(x[1])), reverse=True)[:3]

# Açıklamayı yazıya çevirim
    explanation = f"Predicted: {predicted_disease.upper()}\nTop reasons:\n"
    for symptom, score in top_factors:
        direction = "suggests" if score > 0 else "against"
        explanation += f"  - {symptom} {direction} this diagnosis (score: {score:.3f})\n"

    return predicted_disease, explanation