use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::activation::Activation;

/// Représente une couche du réseau neuronal
#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: Activation,
}

impl Layer {
    /// Crée une nouvelle couche avec initialisation aléatoire
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialisation He/Xavier améliorée
        let std_dev = match activation {
            Activation::Relu => (2.0 / input_size as f64).sqrt(),
            Activation::Sigmoid | Activation::Tanh => (1.0 / input_size as f64).sqrt(),
            Activation::Linear => (1.0 / input_size as f64).sqrt(),
        };

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.r#gen::<f64>() * std_dev * 2.0 - std_dev
        });

        let biases = Array2::zeros((output_size, 1));

        Layer {
            weights,
            biases,
            activation,
        }
    }

    /// Propagation avant à travers cette couche
    pub fn forward(&self, input: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // z = W · input + b
        let z = self.weights.dot(input) + &self.biases;
        
        // a = activation(z)
        let a = self.activation.apply(&z);
        
        (z, a)
    }

    /// Met à jour les poids et biais
    pub fn update_parameters(&mut self, dw: &Array2<f64>, db: &Array2<f64>, learning_rate: f64) {
        self.weights = &self.weights - learning_rate * dw;
        self.biases = &self.biases - learning_rate * db;
    }

    /// Retourne le nombre de neurones dans cette couche
    pub fn size(&self) -> usize {
        self.weights.nrows()
    }
}