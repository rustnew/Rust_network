use csv::Reader;
use ndarray::Array2;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Clone)]
pub struct FinancialData {
    pub encaissements: Vec<f64>,
    pub decaissements: Vec<f64>,
    pub besoins: Vec<f64>,
    pub dates: Vec<String>,
}

#[derive(Debug)]
pub struct NormalizationParams {
    pub min: f64,
    pub max: f64,
}

impl FinancialData {
    /// Charge les données depuis le fichier CSV
    pub fn load_from_csv(file_path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(file_path)?;
        let mut rdr = Reader::from_reader(file);
        
        let mut encaissements = Vec::new();
        let mut decaissements = Vec::new();
        let mut besoins = Vec::new();
        let mut dates = Vec::new();

        for result in rdr.records() {
            let record = result?;
            
            if record.len() >= 5 {
                let date = record[0].to_string();
                let encaissement: f64 = record[2].parse().unwrap_or(0.0);
                let decaissement: f64 = record[3].parse().unwrap_or(0.0);
                let besoin: f64 = record[4].parse().unwrap_or(0.0);
                
                dates.push(date);
                encaissements.push(encaissement);
                decaissements.push(decaissement);
                besoins.push(besoin);
            }
        }

        Ok(FinancialData {
            encaissements,
            decaissements,
            besoins,
            dates,
        })
    }

    /// Divise les données en entraînement (600) et test (157)
    pub fn split_data(&self, train_size: usize) -> (FinancialData, FinancialData) {
        let train_data = FinancialData {
            encaissements: self.encaissements[..train_size].to_vec(),
            decaissements: self.decaissements[..train_size].to_vec(),
            besoins: self.besoins[..train_size].to_vec(),
            dates: self.dates[..train_size].to_vec(),
        };
        
        let test_data = FinancialData {
            encaissements: self.encaissements[train_size..].to_vec(),
            decaissements: self.decaissements[train_size..].to_vec(),
            besoins: self.besoins[train_size..].to_vec(),
            dates: self.dates[train_size..].to_vec(),
        };
        
        (train_data, test_data)
    }

    /// Normalise les données entre 0 et 1
    pub fn normalize(&self) -> (FinancialData, (NormalizationParams, NormalizationParams, NormalizationParams)) {
        let encaissements_params = Self::calculate_normalization_params(&self.encaissements);
        let decaissements_params = Self::calculate_normalization_params(&self.decaissements);
        let besoins_params = Self::calculate_normalization_params(&self.besoins);
        
        let encaissements_norm = Self::normalize_vector(&self.encaissements, &encaissements_params);
        let decaissements_norm = Self::normalize_vector(&self.decaissements, &decaissements_params);
        let besoins_norm = Self::normalize_vector(&self.besoins, &besoins_params);
        
        let normalized_data = FinancialData {
            encaissements: encaissements_norm,
            decaissements: decaissements_norm,
            besoins: besoins_norm,
            dates: self.dates.clone(),
        };
        
        let params = (encaissements_params, decaissements_params, besoins_params);
        
        (normalized_data, params)
    }
    
    fn calculate_normalization_params(data: &[f64]) -> NormalizationParams {
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        NormalizationParams { min, max }
    }
    
    fn normalize_vector(data: &[f64], params: &NormalizationParams) -> Vec<f64> {
        let range = params.max - params.min;
        if range == 0.0 {
            return vec![0.5; data.len()]; // Éviter la division par zéro
        }
        
        data.iter()
            .map(|&x| (x - params.min) / range)
            .collect()
    }
}

/// Prépare les données pour le réseau de neurones (fenêtres temporelles)
pub struct TimeSeriesPreparer {
    pub window_size: usize,
}

impl TimeSeriesPreparer {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
    
    /// Crée les données d'entraînement avec fenêtres temporelles
    pub fn prepare_training_data(&self, data: &FinancialData) -> (Array2<f64>, Array2<f64>) {
        let n_samples = data.encaissements.len() - self.window_size;
        let mut features = Vec::new();
        let mut targets = Vec::new();
        
        for i in 0..n_samples {
            // Créer une fenêtre de 5 jours avec encaissements et decaissements
            for j in 0..self.window_size {
                features.push(data.encaissements[i + j]);
                features.push(data.decaissements[i + j]);
            }
            
            // La cible est le besoin du jour suivant la fenêtre
            targets.push(data.besoins[i + self.window_size]);
        }
        
        let n_features = self.window_size * 2; // 2 variables par jour × 5 jours = 10 entrées
        let x = Array2::from_shape_vec((n_features, n_samples), features)
            .expect("Erreur création matrice features");
        let y = Array2::from_shape_vec((1, n_samples), targets)
            .expect("Erreur création matrice targets");
        
        (x, y)
    }
    
    /// Prépare les données pour la prédiction
    pub fn prepare_prediction_data(&self, data: &FinancialData) -> Array2<f64> {
        let last_window: Vec<f64> = data.encaissements
            .iter()
            .rev()
            .take(self.window_size)
            .chain(data.decaissements.iter().rev().take(self.window_size))
            .cloned()
            .collect();
            
        Array2::from_shape_vec((self.window_size * 2, 1), last_window)
            .expect("Erreur création données prédiction")
    }
}

/// Dénormalise les prédictions
pub fn denormalize_prediction(prediction: f64, params: &NormalizationParams) -> f64 {
    prediction * (params.max - params.min) + params.min
}