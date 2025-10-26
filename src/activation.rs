use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Fonctions d'activation et leurs dérivées
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
    Linear,
}

impl Activation {
    /// Applique la fonction d'activation
    pub fn apply(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Relu => x.mapv(|v| v.max(0.0)),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Linear => x.clone(),
        }
    }

    /// Calcule la dérivée pour la rétropropagation
    pub fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Sigmoid => {
                let sig = self.apply(x);
                &sig * &(1.0 - &sig)
            }
            Activation::Relu => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Tanh => {
                let tanh = self.apply(x);
                1.0 - &tanh * &tanh
            }
            Activation::Linear => Array2::ones(x.dim()),
        }
    }
}