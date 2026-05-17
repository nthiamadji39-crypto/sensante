import pandas as pd
import numpy as np
# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")
# Verifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

#etape 2

from sklearn.preprocessing import LabelEncoder
# Encoder les variables categoriques en nombres
# Le modele ne comprend que des nombres !
le_sexe = LabelEncoder()
le_region = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])
# Definir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
'toux', 'fatigue', 'maux_tete', 'region_encoded']
X = df[feature_cols]
y = df['diagnostic']
print(f"Features : {X.shape}") # (500, 8)
print(f"Cible : {y.shape}") # (500,)

#etape 3.1

from sklearn.model_selection import train_test_split
# 80% pour l'entrainement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
# 20% pour le test
random_state=42,
# Pour avoir les memes resultats a chaque fois
stratify=y
# Garder les memes proportions de diagnostics
)
print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")


#etape 4

from sklearn.ensemble import RandomForestClassifier
# Creer le modele
model = RandomForestClassifier(
n_estimators=100,
# 100 arbres de decision
random_state=42
# Reproductibilite
)
# Entrainer sur les donnees d'entrainement
model.fit(X_train, y_train)
print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")


#etape 5.1

# Predire sur les donnees de test
y_pred = model.predict(X_test)
# Comparer les 10 premieres predictions avec la realite
comparison = pd.DataFrame({
'Vrai diagnostic': y_test.values[:10],
'Prediction': y_pred[:10]
})
print(comparison)


#etape 5.2

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")


#etape 5.3

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)
# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))



#etape 5.4
# Visualiser avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=model.classes_,
yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion- SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee dans figures/confusion_matrix.png")



#etape 6.1

import joblib
import os
# Creer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)
# Serialiser le modele
joblib.dump(model, "models/model.pkl")
# Verifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")


#etape 6.2

# Sauvegarder les encodeurs (indispensables pour les nouvelles donnees)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
# Sauvegarder la liste des features (pour reference)
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("Encodeurs et metadata sauvegardes.")



#etape 7.1

# Simuler ce que fera l'API en Lab 3 :
# Charger le modele DEPUIS LE FICHIER (pas depuis la memoire)
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")




#etape 7.2
# Un nouveau patient arrive au centre de sante de Medina
nouveau_patient = {
'age': 28,
'sexe': 'F',
'temperature': 39.5,
'tension_sys': 110,
'toux': True,
'fatigue': True,
'maux_tete': True,
'region': 'Dakar'
}
# Encoder les valeurs categoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]
# Preparer le vecteur de features
features = [
nouveau_patient['age'],
sexe_enc,
nouveau_patient['temperature'],nouveau_patient['tension_sys'],
int(nouveau_patient['toux']),
int(nouveau_patient['fatigue']),
int(nouveau_patient['maux_tete']),
region_enc
]
# Predire
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()
print(f"\n--- Resultat du pre-diagnostic---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")
print(f"\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:8s} : {proba:.1%} {bar}")


#exo1
importances = model.feature_importances_
# Ligne 191
for name, imp in sorted(zip(feature_cols, importances), 
                        key=lambda x: x[1], reverse=True):
    # Ligne 193 (doit être indentée !)
    print(f" {name:20s} : {imp:.3f}")



#exo2
import pandas as pd
import joblib

# Charger le modèle sérialisé
model = joblib.load("models/model.pkl")

# Créer 3 patients fictifs avec des profils différents
patients_fictifs = pd.DataFrame([
    {
        "age": 18, "sexe_encoded": 0, "temperature": 36.5, "tension_sys": 120,
        "toux": 0, "fatigue": 0, "maux_tete": 0, "region_encoded": 1
    },  # Jeune sans symptômes
    {
        "age": 35, "sexe_encoded": 1, "temperature": 39.5, "tension_sys": 125,
        "toux": 1, "fatigue": 1, "maux_tete": 1, "region_encoded": 2
    },  # Adulte avec forte fièvre
    {
        "age": 70, "sexe_encoded": 0, "temperature": 37.8, "tension_sys": 135,
        "toux": 1, "fatigue": 0, "maux_tete": 0, "region_encoded": 3
    }   # Patient âgé avec toux
])

# Prédire leur diagnostic
predictions = model.predict(patients_fictifs)

# Afficher les résultats
for i, pred in enumerate(predictions, start=1):
    print(f"Patient {i} → Diagnostic prédit : {pred}")
