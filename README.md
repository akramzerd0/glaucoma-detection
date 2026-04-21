# glaucoma-detection


# Détection du glaucome par Deep Learning

## Description

Le glaucome est une maladie oculaire silencieuse qui peut entraîner une perte de vision irréversible si elle n’est pas détectée à temps.

L’objectif de ce projet est de développer un modèle de Deep Learning capable de classifier des images du fond d’œil en deux catégories :
- Normal (NRG)
- Glaucome (RG)

---

## Objectifs

- Détecter le glaucome à partir d’images médicales  
- Construire un modèle basé sur le transfer learning  
- Améliorer la détection précoce grâce à l’IA  

---

## Datasets

### Ancien dataset
- RIM-ONE DL  
- Limité en taille (~485 images)  
- Risque de surapprentissage  

### Dataset final
- EyePACS  
- Plus grand et plus diversifié  
- Meilleure généralisation  

Nous avons migré vers EyePACS pour améliorer les performances du modèle.

---

## Prétraitement des données

Les étapes suivantes ont été appliquées :

- Redimensionnement des images (224x224)  
- Normalisation via `preprocess_input` (adaptée à VGG16)  
- Augmentation des données :
  - rotation  
  - flip  
  - zoom  
  - translation  
  - variation de luminosité  

---

## Modèle utilisé

Nous avons utilisé le modèle pré-entraîné **VGG16** (transfer learning).

### Phase 1 — Freezing
- Gel de toutes les couches convolutionnelles  
- Entraînement uniquement des couches finales  

### Phase 2 — Fine-tuning
- Dégel partiel des dernières couches  
- Ajustement du modèle avec un learning rate faible  

 - Cela permet d’adapter le modèle tout en conservant les connaissances pré-entraînées.

---

## Évaluation

Le modèle a été évalué à l’aide de :
- Accuracy  
- Matrice de confusion  ![Confusion Matrix](matrix.png)
- Analyse des faux positifs / faux négatifs  

---

## Application

Le modèle a été déployé via **Hugging Face Spaces**.
lien : https://huggingface.co/spaces/zakram23/glaucomaaa

Fonctionnalités :
- Upload d’image  
- Prédiction (Glaucome / Normal)  
- Probabilité associée  
- Visualisation (Grad-CAM)

---

## Outils utilisés

- Python  
- TensorFlow / Keras  
- OpenCV  
- GitHub  
- Jira  
- Hugging Face  

---

## Gestion du projet

### Jira
Utilisé pour :
- organiser les tâches (Epics, User Stories)  
- suivre l’avancement  
- structurer le projet avec CRISP-DM  

### GitHub
Utilisé pour :
- versionner le code  
- collaborer en équipe  
- centraliser le projet  

---

## Limites

- Dataset initial limité  
- Risque de biais  
- Modèle non destiné à remplacer un médecin  

---

## Améliorations futures

- Utilisation de GAN pour augmentation avancée  
- Plus de données médicales  
- Optimisation du modèle  

---

## Équipe

- Akram Zerdaoui  
- Sabrina Dahmar  
- Yasmine Taoudiat  

---

## Conclusion

Ce projet montre comment l’intelligence artificielle peut aider à détecter le glaucome de manière précoce, tout en soulignant l’importance des données et du prétraitement.
