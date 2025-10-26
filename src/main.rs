mod activation;
mod layer;
mod network;

use ndarray::Array2;
use activation::Activation;
use network::NeuralNetwork;

fn main() {
    println!("🎯 Réseau de Neurones Basique - Démonstration");
    println!("=============================================\n");

    // 1. Définition de l'architecture
    let architecture = &[
        (2, Activation::Sigmoid),   // Couche d'entrée: 2 neurones
        (100, Activation::Sigmoid),   // Couche cachée: 4 neurones  
        (1, Activation::Sigmoid),   // Couche de sortie: 1 neurone
    ];

    // 2. Création du réseau
    let mut network = NeuralNetwork::new(architecture, 1.0);
    network.print_architecture();

    // 3. Préparation des données (problème XOR)
    let x_train = Array2::from_shape_vec((2, 4), vec![
        0.0, 0.0, 1.0, 1.0,  // Feature 1
        0.0, 1.0, 0.0, 1.0,  // Feature 2
    ]).unwrap();

    let y_train = Array2::from_shape_vec((1, 4), vec![
        0.0, 1.0, 1.0, 0.0,  // Target (XOR)
    ]).unwrap();

    println!("\n📊 Données d'entraînement:");
    println!("Inputs shape: {:?}", x_train.shape());
    println!("Targets shape: {:?}", y_train.shape());
    println!("Targets: {:?}", y_train.row(0));

    // 4. Entraînement du réseau
    println!("\n🔥 Entraînement en cours...");
    let losses = network.train(&x_train, &y_train, 10000);

    // 5. Évaluation finale
    println!("\n📈 Résultats après entraînement:");
    let final_accuracy = network.evaluate(&x_train, &y_train);
    println!("Précision sur le jeu d'entraînement: {:.2}%", final_accuracy * 100.0);

    // 6. Tests de prédiction
    println!("\n🧪 Tests de prédiction:");
    let test_cases = Array2::from_shape_vec((2, 4), vec![
        0.0, 0.0, 1.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
    ]).unwrap();

    let predictions = network.predict(&test_cases);
    
    for i in 0..4 {
        let input1 = test_cases[[0, i]];
        let input2 = test_cases[[1, i]];
        let prediction = predictions[[0, i]];
        let expected = y_train[[0, i]];
        
        println!("  Input: ({}, {}) → Prédiction: {:.4} (attendu: {})", 
                 input1, input2, prediction, expected);
    }

    // 7. Affichage de la courbe de loss
    println!("\n📉 Loss finale: {:.6}", losses[losses.len() - 1]);
    println!("✅ Réseau entraîné avec succès!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let architecture = &[
            (3, Activation::Sigmoid),
            (2, Activation::Sigmoid),
        ];
        
        let network = NeuralNetwork::new(architecture, 0.1);
        assert_eq!(network.layers.len(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let architecture = &[
            (2, Activation::Sigmoid),
            (1, Activation::Sigmoid),
        ];
        
        let network = NeuralNetwork::new(architecture, 0.1);
        let input = Array2::from_shape_vec((2, 1), vec![0.5, 0.5]).unwrap();
        let output = network.forward(&input);
        
        assert_eq!(output.shape(), &[1, 1]);
        assert!(output[[0, 0]] >= 0.0 && output[[0, 0]] <= 1.0);
    }
}