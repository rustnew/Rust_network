
# 🧠 Neural Network from Scratch in Rust

Un **réseau de neurones entièrement codé à la main en Rust 🦀**, sans framework d’apprentissage automatique externe.
Ce projet implémente toutes les étapes de base du machine learning : **initialisation, propagation avant, rétropropagation, mise à jour des poids et évaluation**.

---

## ✨ Sommaire

1. [Introduction](#-introduction)
2. [Architecture du réseau](#-architecture-du-réseau)
3. [Principe de fonctionnement](#-principe-de-fonctionnement)
4. [Formules mathématiques](#-formules-mathématiques)
5. [Structure du code](#-structure-du-code)
6. [Exemple d’utilisation](#-exemple-dutilisation)
7. [Résultats attendus](#-résultats-attendus)
8. [Améliorations possibles](#-améliorations-possibles)

---

## 🚀 Introduction

Ce projet montre comment **implémenter un réseau de neurones (Neural Network)** *from scratch* en **Rust**, en manipulant directement les **poids, biais, activations et gradients**.

Aucun framework comme TensorFlow ou PyTorch n’est utilisé — tout est fait à la main pour une compréhension profonde du **fonctionnement interne d’un réseau de neurones**.

---

## 🧩 Architecture du réseau

Le réseau est composé d’une série de **couches (`Layer`)** reliées séquentiellement :

```
Input Layer  →  Hidden Layers  →  Output Layer
```

Chaque couche contient :

* une **matrice de poids** `W`
* un **vecteur de biais** `b`
* une **fonction d’activation** `f`

Le réseau est défini par la structure suivante :

```rust
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}
```

---

## ⚙️ Principe de fonctionnement

L’apprentissage d’un réseau de neurones se déroule en **trois grandes étapes** :

### 1. 🧮 Propagation avant (Forward Propagation)

Les données d’entrée passent de couche en couche :

[
Z^{(l)} = W^{(l)}A^{(l-1)} + b^{(l)}
]
[
A^{(l)} = f(Z^{(l)})
]

où :

* ( A^{(l)} ) = activations de la couche l
* ( W^{(l)} ) = poids
* ( b^{(l)} ) = biais
* ( f ) = fonction d’activation (ReLU, Sigmoid, etc.)

---

### 2. 📉 Calcul de la perte (Loss)

On mesure l’écart entre la prédiction et la vérité avec la **Mean Squared Error (MSE)** :

[
L = \frac{1}{m} \sum_{i=1}^{m} (y_{pred}^{(i)} - y_{true}^{(i)})^2
]

---

### 3. 🔁 Rétropropagation (Backpropagation)

Le cœur de l’apprentissage : on calcule les **gradients** de la perte par rapport aux poids et biais, puis on les met à jour.

#### Étape 1 – Erreur de sortie

[
\delta^{(L)} = (A^{(L)} - Y) \odot f'(Z^{(L)})
]

#### Étape 2 – Gradients

[
dW^{(l)} = \frac{1}{m} \delta^{(l)} (A^{(l-1)})^T
]
[
db^{(l)} = \frac{1}{m} \sum \delta^{(l)}
]

#### Étape 3 – Propagation de l’erreur

[
\delta^{(l-1)} = (W^{(l)})^T \delta^{(l)} \odot f'(Z^{(l-1)})
]

#### Étape 4 – Mise à jour des paramètres

[
W^{(l)} := W^{(l)} - \eta \cdot dW^{(l)}
]
[
b^{(l)} := b^{(l)} - \eta \cdot db^{(l)}
]

où :

* ( \eta ) = learning rate
* ( \odot ) = produit élément par élément (Hadamard)

---

## 🧠 Structure du code

### 1. `Layer`

Chaque couche contient ses poids, biais et sa fonction d’activation.

```rust
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: Activation,
}
```

Méthodes principales :

* `forward()` → calcule ( Z ) et ( A )
* `update_parameters()` → met à jour les poids et biais :

  ```rust
  pub fn update_parameters(&mut self, dw: &Array2<f64>, db: &Array2<f64>, learning_rate: f64) {
      self.weights = &self.weights - learning_rate * dw;
      self.biases = &self.biases - learning_rate * db;
  }
  ```

---

### 2. `NeuralNetwork`

```rust
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}
```

Méthodes :

* `new()` → construit le réseau à partir d’une architecture donnée
* `forward()` → passe avant complète
* `train_epoch()` → une époque d’entraînement
* `train()` → boucle d’entraînement complète
* `compute_loss()` → calcule la perte MSE
* `evaluate()` → mesure la précision
* `predict()` → fait une prédiction

---

## 📘 Exemple d’utilisation

### Exemple : apprentissage du XOR 🔀

```rust
use ndarray::array;
use neural_network::NeuralNetwork;
use neural_network::activation::Activation;

fn main() {
    let x_train = array![
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0]
    ];

    let y_train = array![
        [0.0, 1.0, 1.0, 0.0]
    ];

    let architecture = [
        (2, Activation::Relu),
        (3, Activation::Relu),
        (1, Activation::Sigmoid),
    ];

    let mut nn = NeuralNetwork::new(&architecture, 0.1);

    nn.print_architecture();

    let losses = nn.train(&x_train, &y_train, 10000);
    println!("Final loss: {:?}", losses.last());

    let predictions = nn.predict(&x_train);
    println!("Predictions:\n{:?}", predictions);
}
```

### Sortie attendue :

```
🧠 Architecture du réseau:
  Couche 1: 2 neurones (Relu)
  Couche 2: 3 neurones (Relu)
  Couche 3: 1 neurone (Sigmoid)

🚀 Début de l'entraînement...
Epoch 0 - Loss: 0.250000
Epoch 1000 - Loss: 0.040121
Epoch 5000 - Loss: 0.007231
Epoch 9999 - Loss: 0.002341
```

---

## 📊 Résultats attendus

Le modèle apprend la fonction XOR :

| Entrée | Sortie attendue | Sortie prédite |
| :----: | :-------------: | :------------: |
| [0, 0] |        0        |      ~0.01     |
| [0, 1] |        1        |      ~0.98     |
| [1, 0] |        1        |      ~0.97     |
| [1, 1] |        0        |      ~0.05     |

---

## 🧮 Fonctions d’activation supportées

| Fonction    | Formule                         | Dérivée                                          |
| ----------- | ------------------------------- | ------------------------------------------------ |
| **Sigmoid** | ( f(x) = \frac{1}{1 + e^{-x}} ) | ( f'(x) = f(x)(1 - f(x)) )                       |
| **ReLU**    | ( f(x) = \max(0, x) )           | ( f'(x) = 1 \text{ si } x > 0, 0 \text{ sinon} ) |
| **Tanh**    | ( f(x) = \tanh(x) )             | ( f'(x) = 1 - \tanh^2(x) )                       |

---

## 🔧 Améliorations possibles

* [ ] Ajouter la régularisation L2 / Dropout
* [ ] Ajouter la normalisation (BatchNorm)
* [ ] Support pour softmax + cross-entropy
* [ ] Sauvegarde / chargement des poids
* [ ] Visualisation des pertes avec `plotters`

---

## 📜 Licence

MIT License © 2025 – Créé par **Martial Wato**

---

Souhaites-tu que je te **génère directement le fichier `README.md` complet** (avec les formules rendues en Markdown mathématique, les titres stylés, et ton nom d’auteur) que tu pourras **coller dans ton dépôt GitHub** ?
Je peux le formater directement pour un rendu professionnel sur GitHub.
