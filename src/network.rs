use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use crate::layer::Layer;
use crate::activation::Activation;

/// Réseau neuronal profond avec gestion des données temps réel
#[derive(Serialize, Deserialize)]
pub struct DeepNeuralNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub input_size: usize,
    pub output_size: usize,
}

impl DeepNeuralNetwork {
    /// Crée un nouveau réseau neuronal profond
    pub fn new(architecture: &[(usize, Activation)], learning_rate: f64) -> Self {
        assert!(architecture.len() >= 3, "Le réseau profond doit avoir au moins 3 couches");
        
        let mut layers = Vec::new();
        let input_size = architecture[0].0;
        let output_size = architecture[architecture.len() - 1].0;
        
        for i in 0..architecture.len() {
            let (size, activation) = &architecture[i];
            let input_size = if i == 0 {
                *size
            } else {
                architecture[i - 1].0
            };
            
            layers.push(Layer::new(input_size, *size, activation.clone()));
        }

        DeepNeuralNetwork {
            layers,
            learning_rate,
            input_size,
            output_size,
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

    /// Propagation avant avec cache pour l'entraînement
    fn forward_with_cache(&self, input: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut activations = Vec::new();
        let mut z_values = Vec::new();
        
        let mut a = input.clone();
        activations.push(a.clone());
        
        for layer in &self.layers {
            let (z, next_a) = layer.forward(&a);
            z_values.push(z);
            activations.push(next_a.clone());
            a = next_a;
        }

        (z_values, activations)
    }

    /// Entraîne le réseau sur une epoch avec régularisation
    pub fn train_epoch(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>, l2_lambda: f64) -> f64 {
        let m = x_train.ncols() as f64;

        // Propagation avant avec cache
        let (z_values, activations) = self.forward_with_cache(x_train);

        // Calcul de la loss avec régularisation L2
        let output = &activations[activations.len() - 1];
        let mut total_loss = self.compute_loss(output, y_train);
        
        // Ajout de la régularisation L2
        if l2_lambda > 0.0 {
            let mut reg_loss = 0.0;
            for layer in &self.layers {
                reg_loss += layer.weights.iter().map(|w| w * w).sum::<f64>();
            }
            total_loss += (l2_lambda / (2.0 * m)) * reg_loss;
        }

        // Rétropropagation
        let last_layer_index = self.layers.len() - 1;
        let mut delta = (output - y_train) * &self.layers[last_layer_index].activation.derivative(&z_values[last_layer_index]);

        // Stocker les gradients pour mise à jour après calculs
        let mut gradients = Vec::new();

        // Calcul des gradients pour toutes les couches
        for i in (0..self.layers.len()).rev() {
            let prev_activation = &activations[i];
            
            let mut dw = delta.dot(&prev_activation.t()) / m;
            let db = delta.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1)) / m;
            
            // Ajout de la régularisation L2 aux gradients des poids
            if l2_lambda > 0.0 {
                dw = &dw + (l2_lambda / m) * &self.layers[i].weights;
            }
            
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

    /// Entraîne le réseau sur plusieurs epochs avec validation
    pub fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>, 
                 x_val: Option<&Array2<f64>>, y_val: Option<&Array2<f64>>, 
                 epochs: usize, l2_lambda: f64) -> (Vec<f64>, Vec<f64>) {
        let mut train_losses = Vec::new();
        let mut val_losses = Vec::new();
        
        println!("🚀 Début de l'entraînement du réseau profond...");
        for epoch in 0..epochs {
            let train_loss = self.train_epoch(x_train, y_train, l2_lambda);
            train_losses.push(train_loss);
            
            // Calcul de la loss de validation si disponible
            if let (Some(x_val), Some(y_val)) = (x_val, y_val) {
                let val_output = self.forward(x_val);
                let val_loss = self.compute_loss(&val_output, y_val);
                val_losses.push(val_loss);
            }
            
            if epoch % 100 == 0 {
                if val_losses.is_empty() {
                    println!("Epoch {} - Train Loss: {:.6}", epoch, train_loss);
                } else {
                    println!("Epoch {} - Train Loss: {:.6}, Val Loss: {:.6}", 
                             epoch, train_loss, val_losses[val_losses.len() - 1]);
                }
            }
        }
        
        (train_losses, val_losses)
    }

    /// Prédiction en temps réel pour une seule entrée
    pub fn predict_single(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size, "Taille d'entrée incorrecte");
        
        let input_array = Array2::from_shape_vec((self.input_size, 1), input.to_vec()).unwrap();
        let output = self.forward(&input_array);
        
        output.iter().cloned().collect()
    }

    /// Prédiction par lot pour plusieurs entrées
    pub fn predict_batch(&self, inputs: &Array2<f64>) -> Array2<f64> {
        self.forward(inputs)
    }

    /// Calcule l'erreur quadratique moyenne
    pub fn compute_loss(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        let diff = y_pred - y_true;
        (&diff * &diff).mean().unwrap()
    }

    /// Évalue la précision sur un jeu de données
    pub fn evaluate(&self, x_test: &Array2<f64>, y_test: &Array2<f64>) -> f64 {
        let predictions = self.predict_batch(x_test);
        let predicted_classes = predictions.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 });
        
        let correct = predicted_classes.iter()
            .zip(y_test.iter())
            .filter(|(pred, true_val)| (**pred - **true_val).abs() < 0.5)
            .count();
            
        correct as f64 / y_test.len() as f64
    }

    /// Sauvegarde le modèle dans un fichier
    pub fn save(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(file_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        println!("💾 Modèle sauvegardé: {}", file_path);
        Ok(())
    }

    /// Charge le modèle depuis un fichier
    pub fn load(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let model: DeepNeuralNetwork = serde_json::from_reader(reader)?;
        println!("📂 Modèle chargé: {}", file_path);
        Ok(model)
    }

    /// Affiche l'architecture détaillée
    pub fn print_architecture(&self) {
        println!("🧠 Architecture du Réseau Profond:");
        println!("  Couches: {}", self.layers.len());
        println!("  Taille d'entrée: {}", self.input_size);
        println!("  Taille de sortie: {}", self.output_size);
        
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_type = if i == 0 {
                "Entrée"
            } else if i == self.layers.len() - 1 {
                "Sortie"
            } else {
                "Cachée"
            };
            
            println!("  {} {}: {} neurones, Activation: {:?}, Poids: {:?}", 
                     layer_type, i + 1, 
                     layer.size(),
                     layer.activation,
                     layer.weights.shape());
        }
    }
}