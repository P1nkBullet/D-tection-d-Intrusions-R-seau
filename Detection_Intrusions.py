import pandas as pd

print("********ÉTAPE 1 : COLLECTE DES DONNÉES : ********")
dataset = pd.read_csv("/Users/jihane/Downloads/OneDrive_1_11-6-2026/UNSW_NB15_training-set.csv",sep=",")
print(dataset.shape)
print(dataset.head())
print(dataset.info())  
print(dataset.nunique()) 
print("\n")

print("******** ÉTAPE 2 : NETTOYAGE DES DONNÉES :********")

print("**************** Suppression des colonnes : ****************")
dataset = dataset.drop(columns=["id","attack_cat"])
print(dataset.shape)

print("**************** Suppression des lignes :   ****************")
print("****DOUBLONS : ")
print("On vérifie s'il y a des doublons dans le dataset :",dataset.duplicated().sum())
dataset = dataset.drop_duplicates()
print("On vérifie s'il y a des doublons dans le dataset après suppression :",dataset.duplicated().sum())
print(dataset.shape)

print("****VALEURS MANQUANTES : ")
print(dataset.isnull().sum())
#aucune valeur manquante dans le dataset

print("****VALEURS ABERRANTES : ")
print(dataset.describe())
colonnes_num = dataset.select_dtypes(include="number").columns.drop("label")

for col in colonnes_num:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    print("[",limite_inf,",",limite_sup,"]")
    VA = (dataset[col] < limite_inf) | (dataset[col] > limite_sup)
    print(col, ":", VA.sum(), "valeurs aberrantes.")
    #On garde les valeurs aberrantes car elles pourraient être des attaques réelles et nous ne voulons pas les supprimer.

print("********Etape 3 : Séparation X / y ********")
y = dataset["label"]
X = dataset.drop(columns=["label"])
print("X :", X.shape, " y :", y.shape)

print("******** ÉTAPE 4 : TRANSFORMATION :********")

print("**** 4.1 Encodage one-hot ****")
colonnes_texte = X.select_dtypes(include="str").columns
print("colonnes texte :", list(colonnes_texte))
for col in colonnes_texte:
    print(col, ":", X[col].nunique(), "valeurs différentes")

X = pd.get_dummies(X, columns=colonnes_texte, dtype=int)
print("dimensions après encodage :", X.shape)

print("**** 4.2 Normalisation ****")
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
print("min :", X_norm.min(), " max :", X_norm.max())

print("**** 4.3 Réduction de dimension (PCA) ****")
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_norm)
print("colonnes avant PCA :", X_norm.shape[1])
print("colonnes après PCA :", X_pca.shape[1])
print("information conservée :", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

print("########## Vérification finale ##########")
print("Dimension de X_pca :", X_pca.shape)
print("Dimension de y :", y.shape)
print("Valeurs manquantes :", pd.isnull(X_pca).sum())


#On va utiliser les deux méthodes pour comparer 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model_rl = LogisticRegression(max_iter=1000)
model_rl.fit(X_pca, y)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_pca, y)

print("Les deux modèles sont entraînés")


print("******** ÉTAPE 5 : PRÉPARATION DU FICHIER TEST ********")
test = pd.read_csv("/Users/jihane/Downloads/OneDrive_1_11-6-2026/UNSW_NB15_testing-set.csv", sep=",")
test = test.drop(columns=["id", "attack_cat"])
y_test = test["label"]
X_test = test.drop(columns=["label"])

X_test = pd.get_dummies(X_test, columns=colonnes_texte, dtype=int)
X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_test_norm = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_norm)
print("X_test_pca :", X_test_pca.shape, " y_test :", y_test.shape)


print("******** ÉTAPE 6 : ÉVALUATION ********")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

for nom, modele in [("Régression logistique", model_rl), ("Random Forest", model_rf)]:
    y_pred = modele.predict(X_test_pca)
    print("\n=====", nom, "=====")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


print("******** ÉTAPE 7 : MÉTHODE HYBRIDE (RF -> top-k -> LR) ********")
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

Xn = pd.DataFrame(X_norm, columns=X.columns)
Xn_test = pd.DataFrame(X_test_norm, columns=X.columns)

X_tr, X_val, y_tr, y_val = train_test_split(Xn, y, test_size=0.2, random_state=42, stratify=y)

rf_val = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
importances = pd.Series(rf_val.feature_importances_, index=Xn.columns)

meilleur_k, meilleur_f1 = None, -1
for k in [10, 20, 30, 50, 100, Xn.shape[1]]:
    top = importances.nlargest(k).index
    m = LogisticRegression(max_iter=1000).fit(X_tr[top], y_tr)
    f1 = f1_score(y_val, m.predict(X_val[top]))
    print("k =", k, " F1 validation =", round(f1, 4))
    if f1 > meilleur_f1:
        meilleur_k, meilleur_f1 = k, f1
print("Meilleur k :", meilleur_k)

top_final = importances.nlargest(meilleur_k).index
model_hyb = LogisticRegression(max_iter=1000).fit(Xn[top_final], y)
y_pred = model_hyb.predict(Xn_test[top_final])

print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Précision:", round(precision_score(y_test, y_pred), 3))
print("Rappel   :", round(recall_score(y_test, y_pred), 3))
print("F1-score :", round(f1_score(y_test, y_pred), 3))
print(confusion_matrix(y_test, y_pred))



