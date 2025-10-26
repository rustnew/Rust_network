mod activation;
mod layer;
mod network;
mod data_processor;

use ndarray::Array2;
use activation::Activation;
use network::DeepNeuralNetwork;
use data_processor::{FinancialData, TimeSeriesPreparer, denormalize_prediction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 RÉSEAU DE NEURONES PROFOND - PRÉDICTION FINANCIÈRE");
    println!("=====================================================\n");

    // 1. CHARGEMENT DES DONNÉES RÉELLES
    let data_path = "data/Agence_00001.csv";
    println!("📂 Chargement des données depuis: {}", data_path);
    
    let raw_data = FinancialData::load_from_csv(data_path)?;
    println!("   Données chargées: {} lignes", raw_data.encaissements.len());

    // Aperçu des données
    println!("\n📊 Aperçu des données réelles:");
    for i in 0..3.min(raw_data.dates.len()) {
        println!("   {}: Enc={:.0}, Dec={:.0}, Besoin={:.0}", 
                 raw_data.dates[i], 
                 raw_data.encaissements[i], 
                 raw_data.decaissements[i], 
                 raw_data.besoins[i]);
    }

    // 2. DIVISION ENTRAÎNEMENT/TEST (600 pour l'entraînement, le reste pour test)
    println!("\n📊 Division des données...");
    let train_size = 600;
    let (train_data_raw, test_data_raw) = raw_data.split_data(train_size);
    println!("   Entraînement: {} lignes", train_data_raw.encaissements.len());
    println!("   Test: {} lignes", test_data_raw.encaissements.len());

    // 3. NORMALISATION
    println!("\n🔧 Normalisation des données...");
    let (train_data_norm, train_params) = train_data_raw.normalize();
    let (test_data_norm, _) = test_data_raw.normalize();
    
    let (encaissements_params, decaissements_params, besoins_params) = train_params;

    // 4. PRÉPARATION DES DONNÉES TEMPORELLES
    println!("\n⏰ Création des fenêtres temporelles...");
    let preparer = TimeSeriesPreparer::new(5); // Fenêtre de 5 jours → 10 entrées
    
    let (x_train, y_train) = preparer.prepare_training_data(&train_data_norm);
    let (x_test, y_test) = preparer.prepare_training_data(&test_data_norm);
    
    println!("   Données d'entraînement: {:?}", x_train.shape());
    println!("   Données de test: {:?}", x_test.shape());

    // 5. CRÉATION DU RÉSEAU PROFOND (MÊME ARCHITECTURE)
    let architecture = &[
        (10, Activation::Relu),      // Couche d'entrée: 10 neurones (5 jours × 2 variables)
        (32, Activation::Relu),      // Couche cachée 1
        (64, Activation::Relu),      // Couche cachée 2  
        (128, Activation::Relu),     // Couche cachée 3
        (64, Activation::Relu),      // Couche cachée 4
        (32, Activation::Relu),      // Couche cachée 5
        (1, Activation::Linear),     // Couche de sortie: besoin prédit (régression)
    ];

    let mut deep_net = DeepNeuralNetwork::new(architecture, 0.01);
    deep_net.print_architecture();

    // 6. ENTRAÎNEMENT DU RÉSEAU PROFOND AVEC RÉGULARISATION
    println!("\n🔥 Entraînement du réseau profond...");
    let (train_losses, val_losses) = deep_net.train(
        &x_train, 
        &y_train,
        Some(&x_test), 
        Some(&y_test),
        2000,  // Plus d'epochs pour le réseau profond
        0.001  // Régularisation L2
    );

    // 7. SAUVEGARDE DU MODÈLE
    deep_net.save("financial_deep_model.json")?;
    println!("💾 Modèle sauvegardé: financial_deep_model.json");

    // 8. DÉMONSTRATION DE PRÉDICTION EN TEMPS RÉEL
    println!("\n⏱️  Démonstration Temps Réel:");
    
    // Préparer les données pour la prédiction du prochain jour
    let prediction_data = preparer.prepare_prediction_data(&test_data_norm);
    let prediction_norm = deep_net.predict_batch(&prediction_data);
    let prediction_actual = denormalize_prediction(prediction_norm[[0, 0]], &besoins_params);
    
    println!("   📍 Fichier source: {}", data_path);
    println!("   📅 Dernière date connue: {}", test_data_raw.dates.last().unwrap_or(&"N/A".to_string()));
    println!("   💰 Besoin prédit pour le prochain jour: {:.2} XOF", prediction_actual);

    // 9. TESTS SUPPLÉMENTAIRES AVEC D'AUTRES FENÊTRES
    println!("\n🧪 Tests supplémentaires:");
    let test_cases = prepare_test_cases(&test_data_norm, &preparer);
    for (i, test_input) in test_cases.iter().enumerate() {
        let prediction_norm = deep_net.predict_batch(test_input);
        let prediction_actual = denormalize_prediction(prediction_norm[[0, 0]], &besoins_params);
        println!("   Test {}: Prédiction = {:.2} XOF", i + 1, prediction_actual);
    }

    // 10. ÉVALUATION FINALE
    println!("\n📈 Performances Finales:");
    
    // Prédictions sur l'ensemble de test
    let test_predictions = deep_net.predict_batch(&x_test);
    let test_loss = deep_net.compute_loss(&test_predictions, &y_test);
    
    // Calcul de la précision pour la classification binaire (si on veut)
    let train_predictions = deep_net.predict_batch(&x_train);
    let train_accuracy = calculate_regression_accuracy(&train_predictions, &y_train, &besoins_params);
    let test_accuracy = calculate_regression_accuracy(&test_predictions, &y_test, &besoins_params);
    
    println!("   Loss finale - Entraînement: {:.6}", train_losses[train_losses.len() - 1]);
    println!("   Loss finale - Test: {:.6}", test_loss);
    println!("   Précision relative - Entraînement: {:.2}%", train_accuracy * 100.0);
    println!("   Précision relative - Test: {:.2}%", test_accuracy * 100.0);

    if let Some(last_val_loss) = val_losses.last() {
        println!("   Loss validation: {:.6}", last_val_loss);
    }

    // 11. CHARGEMENT DU MODÈLE POUR DÉMONSTRATION
    println!("\n🔄 Test de chargement du modèle...");
    let loaded_net = DeepNeuralNetwork::load("financial_deep_model.json")?;
    let loaded_prediction = loaded_net.predict_batch(&prediction_data);
    let loaded_prediction_actual = denormalize_prediction(loaded_prediction[[0, 0]], &besoins_params);
    println!("   Prédiction après chargement: {:.2} XOF", loaded_prediction_actual);

    println!("\n✅ RÉSEAU PROFOND ENTRAÎNÉ SUR DONNÉES RÉELLES AVEC SUCCÈS!");
    println!("   Fichier: {}", data_path);
    println!("   Architecture: 10 → 32 → 64 → 128 → 64 → 32 → 1");
    println!("   Prédictions générées avec succès!");

    Ok(())
}

/// Prépare des cas de test supplémentaires
fn prepare_test_cases(test_data: &FinancialData, preparer: &TimeSeriesPreparer) -> Vec<Array2<f64>> {
    let mut test_cases = Vec::new();
    
    // Prendre plusieurs fenêtres différentes pour tester
    let test_indices = vec![
        test_data.encaissements.len() - 10, // Il y a 10 jours
        test_data.encaissements.len() - 15, // Il y a 15 jours  
        test_data.encaissements.len() - 20, // Il y a 20 jours
    ];
    
    for &start_idx in &test_indices {
        if start_idx >= preparer.window_size {
            let window_data = FinancialData {
                encaissements: test_data.encaissements[start_idx - preparer.window_size..start_idx].to_vec(),
                decaissements: test_data.decaissements[start_idx - preparer.window_size..start_idx].to_vec(),
                besoins: test_data.besoins[start_idx - preparer.window_size..start_idx].to_vec(),
                dates: test_data.dates[start_idx - preparer.window_size..start_idx].to_vec(),
            };
            
            let test_case = preparer.prepare_prediction_data(&window_data);
            test_cases.push(test_case);
        }
    }
    
    test_cases
}

/// Calcule une précision relative pour la régression
fn calculate_regression_accuracy(predictions: &Array2<f64>, targets: &Array2<f64>, params: &data_processor::NormalizationParams) -> f64 {
    let range = params.max - params.min;
    if range == 0.0 {
        return 0.0;
    }
    
    let mut correct_predictions = 0.0;
    let total_predictions = predictions.len() as f64;
    
    for i in 0..predictions.len() {
        let pred = predictions[[0, i]];
        let target = targets[[0, i]];
        let error = (pred - target).abs();
        
        // Considérer comme correct si l'erreur est inférieure à 10% de la plage
        if error < 0.1 {
            correct_predictions += 1.0;
        }
    }
    
    correct_predictions / total_predictions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deep_network_creation() {
        let architecture = &[
            (10, Activation::Relu),
            (32, Activation::Relu),
            (64, Activation::Relu),
            (128, Activation::Relu),
            (64, Activation::Relu),
            (32, Activation::Relu),
            (1, Activation::Linear),
        ];
        
        let net = DeepNeuralNetwork::new(architecture, 0.1);
        assert_eq!(net.layers.len(), 7);
        assert_eq!(net.input_size, 10);
        assert_eq!(net.output_size, 1);
    }

    #[test]
    fn test_data_loading() {
        // Test que le chargement des données fonctionne
        // Note: Ce test nécessite que le fichier de données existe
        let result = FinancialData::load_from_csv("data/Agence_00001.csv");
        assert!(result.is_ok(), "Le chargement des données devrait fonctionner");
    }
}