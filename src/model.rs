use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::{
    data::SurvivalData,
    error::{CoxError, Result},
    optimization::{CoxOptimizer, OptimizationConfig},
};

/// cox model w/ elastic net regularization
#[derive(Debug, Clone)]  
pub struct CoxModel {
    coefficients: Option<Array1<f64>>,  // fitted coefficients
    l1_penalty: f64,                    // lasso penalty 
    l2_penalty: f64,                    // ridge penalty
    max_iterations: usize,              // optimization limit
    tolerance: f64,                     // convergence threshold
    fitted: bool,                       // have we been fit yet?
    feature_names: Option<Vec<String>>, // optional feature labels
}

impl Default for CoxModel {
    fn default() -> Self {
        Self {
            coefficients: None,
            l1_penalty: 0.0,
            l2_penalty: 0.0,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted: false,
            feature_names: None,
        }
    }
}

impl CoxModel {
    /// new cox model w/ defaults
    pub fn new() -> Self {
        Self::default()
    }
    
    /// add lasso penalty (L1) - encourages sparsity 
    pub fn with_l1_penalty(mut self, penalty: f64) -> Self {
        self.l1_penalty = penalty.max(0.0);
        self
    }
    
    /// add ridge penalty (L2) - shrinks coefficients
    pub fn with_l2_penalty(mut self, penalty: f64) -> Self {
        self.l2_penalty = penalty.max(0.0);
        self
    }
    
    /// elastic net mixing: alpha=0 -> pure ridge, alpha=1 -> pure lasso
    pub fn with_elastic_net(mut self, alpha: f64, penalty: f64) -> Self {
        if alpha < 0.0 || alpha > 1.0 {
            panic!("alpha must be in [0,1]");
        }
        self.l1_penalty = alpha * penalty;        // lasso component
        self.l2_penalty = (1.0 - alpha) * penalty; // ridge component  
        self
    }
    
    /// max iterations before giving up
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }
    
    /// how close is close enough for convergence
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    
    /// give names to your features for nicer output
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }
    
    /// fit the model to data - this does the actual work
    pub fn fit(&mut self, data: &SurvivalData) -> Result<&mut Self> {
        let config = OptimizationConfig {
            l1_penalty: self.l1_penalty,
            l2_penalty: self.l2_penalty,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        };
        
        let optimizer = CoxOptimizer::new(config);
        self.coefficients = Some(optimizer.optimize(data)?);
        self.fitted = true;
        
        Ok(self)
    }
    
    /// get the fitted coefficients (betas)
    pub fn coefficients(&self) -> Result<ArrayView1<'_, f64>> {
        match &self.coefficients {
            Some(coefs) => Ok(coefs.view()),
            None => Err(CoxError::ModelNotFitted),
        }
    }
    
    /// predict risk scores for new patients  
    pub fn predict(&self, covariates: ArrayView2<f64>) -> Result<Array1<f64>> {
        let coefs = self.coefficients()?;
        
        if covariates.ncols() != coefs.len() {
            return Err(CoxError::invalid_dimensions(
                format!("feature count mismatch: expected {}, got {}", 
                       coefs.len(), covariates.ncols())
            ));
        }
        
        Ok(covariates.dot(&coefs))  // linear combination
    }
    
    /// predict hazard ratios (exp of risk scores)
    pub fn predict_hazard_ratios(&self, covariates: ArrayView2<f64>) -> Result<Array1<f64>> {
        let linear_predictors = self.predict(covariates)?;
        Ok(linear_predictors.mapv(f64::exp))
    }
    
    /// predict survival probs at specific time points  
    pub fn predict_survival(&self, covariates: ArrayView2<f64>, times: ArrayView1<f64>) -> Result<Array2<f64>> {
        let risk_scores = self.predict(covariates)?;
        let n_samples = covariates.nrows();
        let n_times = times.len();
        
        // simplified survival estimation (in practice use breslow estimator)
        let mut survival_probs = Array2::zeros((n_samples, n_times));
        
        for (i, &time) in times.iter().enumerate() {
            for j in 0..n_samples {
                let hazard_ratio = risk_scores[j].exp();
                let baseline_hazard = 0.1; // rough approximation 
                survival_probs[[j, i]] = (-baseline_hazard * hazard_ratio * time).exp();
            }
        }
        
        Ok(survival_probs)
    }
    
    /// feature importance = abs value of coefficients
    pub fn feature_importance(&self) -> Result<Array1<f64>> {
        let coefs = self.coefficients()?;
        Ok(coefs.mapv(f64::abs))
    }
    
    /// get a nice summary of the fitted model
    pub fn summary(&self) -> Result<CoxModelSummary> {
        if !self.fitted {
            return Err(CoxError::ModelNotFitted);
        }
        
        let coefs = self.coefficients()?.to_owned();
        let hazard_ratios = coefs.mapv(f64::exp);
        
        Ok(CoxModelSummary {
            coefficients: coefs,
            hazard_ratios,
            l1_penalty: self.l1_penalty,
            l2_penalty: self.l2_penalty,
            feature_names: self.feature_names.clone(),
        })
    }
    
    /// has this model been fit to data yet?
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
    
    /// what regularization penalties are we using?
    pub fn regularization_params(&self) -> (f64, f64) {
        (self.l1_penalty, self.l2_penalty)  // (lasso, ridge)
    }
}

/// nice summary of what the model learned
#[derive(Debug, Clone)]
pub struct CoxModelSummary {
    pub coefficients: Array1<f64>,   // the betas  
    pub hazard_ratios: Array1<f64>,  // exp(betas)
    pub l1_penalty: f64,             // lasso penalty used
    pub l2_penalty: f64,             // ridge penalty used
    pub feature_names: Option<Vec<String>>, // optional labels
}

impl CoxModelSummary {
    /// print out what we learned
    pub fn print(&self) {
        println!("cox proportional hazards model summary");
        println!("=====================================");
        println!("l1 penalty (lasso): {:.6}", self.l1_penalty);
        println!("l2 penalty (ridge): {:.6}", self.l2_penalty);
        println!("");
        
        println!("{:<20} {:>12} {:>12}", "feature", "coefficient", "hazard ratio");
        println!("{:-<44}", "");
        
        for i in 0..self.coefficients.len() {
            let default_name = format!("x{}", i);
            let feature_name = match &self.feature_names {
                Some(names) => names.get(i).map(|s| s.as_str()).unwrap_or(&default_name),
                None => &default_name,
            };
            
            println!("{:<20} {:>12.6} {:>12.6}", 
                    feature_name,
                    self.coefficients[i],
                    self.hazard_ratios[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use approx::assert_relative_eq;
    
    fn create_test_data() -> SurvivalData {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true, false, true, true, false, true, true, false];
        let covariates = Array2::from_shape_vec((8, 3), vec![
            1.0, 0.0, 0.5,
            0.0, 1.0, -0.5,
            1.0, 1.0, 0.0,
            -1.0, 0.0, 1.0,
            0.0, -1.0, -1.0,
            1.0, -1.0, 0.5,
            -1.0, 1.0, -0.5,
            0.0, 0.0, 0.0,
        ]).unwrap();
        
        SurvivalData::new(times, events, covariates).unwrap()
    }
    
    #[test]
    fn test_model_creation() {
        let model = CoxModel::new()
            .with_l1_penalty(0.1)
            .with_l2_penalty(0.05)
            .with_max_iterations(500);
        
        assert_eq!(model.l1_penalty, 0.1);
        assert_eq!(model.l2_penalty, 0.05);
        assert_eq!(model.max_iterations, 500);
        assert!(!model.is_fitted());
    }
    
    #[test]
    fn test_elastic_net_parameters() {
        let model = CoxModel::new().with_elastic_net(0.5, 1.0);
        assert_relative_eq!(model.l1_penalty, 0.5, epsilon = 1e-10);
        assert_relative_eq!(model.l2_penalty, 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_model_not_fitted_error() {
        let model = CoxModel::new();
        assert!(model.coefficients().is_err());
        assert!(model.summary().is_err());
        
        let covariates = Array2::zeros((5, 3));
        assert!(model.predict(covariates.view()).is_err());
    }
    
    #[test]
    fn test_feature_names() {
        let names = vec!["age".to_string(), "gender".to_string(), "treatment".to_string()];
        let model = CoxModel::new().with_feature_names(names.clone());
        assert_eq!(model.feature_names.unwrap(), names);
    }
    
    #[test]
    fn test_prediction_dimension_mismatch() {
        let data = create_test_data();
        let mut model = CoxModel::new();
        model.fit(&data).unwrap();
        
        // Wrong number of features
        let wrong_covariates = Array2::zeros((5, 2)); // Should be 3 features
        assert!(model.predict(wrong_covariates.view()).is_err());
    }
}