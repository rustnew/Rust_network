use ndarray::Array2;
use crate::layer::Layer;
use crate::activation::Activation;

/// Réseau neuronal basique avec entraînement intégré
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}

impl NeuralNetwork {
    /// Crée un nouveau réseau neuronal
    pub fn new(architecture: &[(usize, Activation)], learning_rate: f64) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..architecture.len() {
            let (size, activation) = &architecture[i];
            let input_size = if i == 0 {
                *size
            } else {
                architecture[i - 1].0
            };
            
            layers.push(Layer::new(input_size, *size, activation.clone()));
        }

        NeuralNetwork {
            layers,
            learning_rate,
        }
    }

    /// Propagation avant complète
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut a = input.clone();
        
        for layer in &self.layers {
            let (_, next_a) = layer.forward(&a);
            a = next_a;
        }
        
        a
    }

    /// Entraîne le réseau sur une epoch
    pub fn train_epoch(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> f64 {
        let m = x_train.ncols() as f64;

        // Propagation avant
        let mut activations = Vec::new();
        let mut z_values = Vec::new();
        
        let mut a = x_train.clone();
        activations.push(a.clone());
        
        for layer in &self.layers {
            let (z, next_a) = layer.forward(&a);
            z_values.push(z);
            activations.push(next_a.clone());
            a = next_a;
        }

        // Calcul de la loss
        let output = &activations[activations.len() - 1];
        let total_loss = self.compute_loss(output, y_train);

        // Rétropropagation - couche de sortie
        let last_layer_index = self.layers.len() - 1;
        let mut delta = (output - y_train) * &self.layers[last_layer_index].activation.derivative(&z_values[last_layer_index]);

        // Stocker les gradients pour mise à jour après calculs
        let mut gradients = Vec::new();

        // Calcul des gradients pour toutes les couches
        for i in (0..self.layers.len()).rev() {
            let prev_activation = &activations[i];
            
            let dw = delta.dot(&prev_activation.t()) / m;
            let db = delta.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1)) / m;
            
            gradients.push((i, dw, db));
            
            // Propagation de l'erreur vers l'arrière (sauf pour la première couche)
            if i > 0 {
                delta = self.layers[i].weights.t().dot(&delta) * 
                       &self.layers[i-1].activation.derivative(&z_values[i-1]);
            }
        }

        // Mise à jour des paramètres après tous les calculs
        for (i, dw, db) in gradients {
            self.layers[i].update_parameters(&dw, &db, self.learning_rate);
        }

        total_loss
    }

    /// Entraîne le réseau sur plusieurs epochs
    pub fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>, epochs: usize) -> Vec<f64> {
        let mut losses = Vec::new();
        
        println!("🚀 Début de l'entraînement...");
        for epoch in 0..epochs {
            let loss = self.train_epoch(x_train, y_train);
            losses.push(loss);
            
            if epoch % 100 == 0 {
                println!("Epoch {} - Loss: {:.6}", epoch, loss);
            }
        }
        
        losses
    }

    /// Calcule l'erreur quadratique moyenne
    pub fn compute_loss(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        let diff = y_pred - y_true;
        (&diff * &diff).mean().unwrap()
    }

    /// Prédit et retourne les classes
    pub fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        self.forward(input)
    }

    /// Évalue la précision sur un jeu de données
    pub fn evaluate(&self, x_test: &Array2<f64>, y_test: &Array2<f64>) -> f64 {
        let predictions = self.predict(x_test);
        let predicted_classes = predictions.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 });
        
        let correct = predicted_classes.iter()
            .zip(y_test.iter())
            .filter(|(pred, true_val)| (**pred - **true_val).abs() < 0.5)
            .count();
            
        correct as f64 / y_test.len() as f64
    }

    /// Affiche l'architecture
    pub fn print_architecture(&self) {
        println!("🧠 Architecture du réseau:");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("  Couche {}: {} neurones ({:?})", 
                i + 1, 
                layer.weights.nrows(),
                layer.activation
            );
        }
    }
}