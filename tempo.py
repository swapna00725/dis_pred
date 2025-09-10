import pickle

with open("artifacts/disease_predictor.pkl", "rb") as f:
    data = pickle.load(f)

print("✅ Keys in pickle:", data.keys())