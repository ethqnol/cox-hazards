use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::error::{CoxError, Result};

/// survival data - times, events, and patient features
#[derive(Debug, Clone)]
pub struct SurvivalData {
    times: Array1<f64>,              // time to event/censoring
    events: Array1<bool>,            // true = event, false = censored  
    covariates: Array2<f64>,         // patient features (n_samples x n_features)
    risk_set_indices: Vec<Vec<usize>>, // precomputed for efficiency
}

impl SurvivalData {
    /// make new survival data from raw vecs/arrays
    pub fn new(
        times: Vec<f64>,        // survival/censoring times  
        events: Vec<bool>,      // true = event occurred, false = censored
        covariates: Array2<f64>, // patient features matrix
    ) -> Result<Self> {
        let n_samples = times.len();
        
        if events.len() != n_samples {
            return Err(CoxError::invalid_dimensions(
                format!("times len ({}) != events len ({})", n_samples, events.len())
            ));
        }
        
        if covariates.nrows() != n_samples {
            return Err(CoxError::invalid_dimensions(
                format!("covariates rows ({}) != n_samples ({})", covariates.nrows(), n_samples)
            ));
        }
        
        if times.iter().any(|&t| t <= 0.0 || !t.is_finite()) {
            return Err(CoxError::invalid_survival_data(
                "survival times must be positive & finite"
            ));
        }
        
        let times = Array1::from(times);
        let events = Array1::from(events);
        
        let mut data = Self {
            times,
            events,
            covariates,
            risk_set_indices: Vec::new(),
        };
        
        data.compute_risk_sets()?;
        Ok(data)
    }
    
    /// precompute risk sets for each event time (who's still at risk)
    fn compute_risk_sets(&mut self) -> Result<()> {
        let mut event_times: Vec<f64> = self.times
            .iter()
            .zip(self.events.iter())
            .filter_map(|(time, event)| if *event { Some(*time) } else { None })
            .collect();
        
        event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        event_times.dedup();  // remove duplicate event times
        
        self.risk_set_indices.clear();
        
        for &event_time in &event_times {
            // everyone who hasn't died/been censored before this time
            let risk_set: Vec<usize> = (0..self.n_samples())
                .filter(|&i| self.times[i] >= event_time)
                .collect();
            self.risk_set_indices.push(risk_set);
        }
        
        Ok(())
    }
    
    /// how many patients
    pub fn n_samples(&self) -> usize {
        self.times.len()
    }
    
    /// how many features per patient  
    pub fn n_features(&self) -> usize {
        self.covariates.ncols()
    }
    
    /// survival/censoring times
    pub fn times(&self) -> ArrayView1<'_, f64> {
        self.times.view()
    }
    
    /// event indicators (true = event, false = censored)
    pub fn events(&self) -> &[bool] {
        self.events.as_slice().unwrap()
    }
    
    /// patient feature matrix
    pub fn covariates(&self) -> ArrayView2<'_, f64> {
        self.covariates.view()
    }
    
    /// precomputed risk sets for optimization
    pub fn risk_sets(&self) -> &[Vec<usize>] {
        &self.risk_set_indices
    }
    
    /// unique event times in order
    pub fn event_times(&self) -> Vec<f64> {
        let mut times: Vec<f64> = self.times
            .iter()
            .zip(self.events.iter())
            .filter_map(|(time, event)| if *event { Some(*time) } else { None })
            .collect();
        
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times.dedup();
        times
    }
    
    /// grab a subset of patients by indices
    pub fn subset(&self, indices: &[usize]) -> Result<Self> {
        if indices.iter().any(|&i| i >= self.n_samples()) {
            return Err(CoxError::invalid_dimensions(
                "subset index out of bounds"
            ));
        }
        
        let times: Vec<f64> = indices.iter().map(|&i| self.times[i]).collect();
        let events: Vec<bool> = indices.iter().map(|&i| self.events[i]).collect();
        let covariates = self.covariates.select(ndarray::Axis(0), indices);
        
        Self::new(times, events, covariates)
    }
    
    /// standardize features (mean=0, std=1) - modifies in place
    pub fn standardize_covariates(&mut self) -> Result<(Array1<f64>, Array1<f64>)> {
        let means = self.covariates.mean_axis(ndarray::Axis(0)).unwrap();
        let stds = self.covariates.std_axis(ndarray::Axis(0), 0.0);
        
        for j in 0..self.n_features() {
            if stds[j] == 0.0 {
                return Err(CoxError::numerical_error(
                    format!("feature {} has zero variance - can't standardize", j)
                ));
            }
            
            // z-score normalization  
            for i in 0..self.n_samples() {
                self.covariates[[i, j]] = (self.covariates[[i, j]] - means[j]) / stds[j];
            }
        }
        
        Ok((means, stds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    fn create_test_data() -> SurvivalData {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, false, true, true, false];
        let covariates = Array2::from_shape_vec((5, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
            9.0, 10.0,
        ]).unwrap();
        
        SurvivalData::new(times, events, covariates).unwrap()
    }
    
    #[test]
    fn test_survival_data_creation() {
        let data = create_test_data();
        assert_eq!(data.n_samples(), 5);
        assert_eq!(data.n_features(), 2);
        assert_eq!(data.event_times(), vec![1.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_invalid_dimensions() {
        let times = vec![1.0, 2.0];
        let events = vec![true];  // Wrong length
        let covariates = Array2::zeros((2, 2));
        
        assert!(SurvivalData::new(times, events, covariates).is_err());
    }
    
    #[test]
    fn test_invalid_times() {
        let times = vec![-1.0, 2.0];  // Negative time
        let events = vec![true, false];
        let covariates = Array2::zeros((2, 2));
        
        assert!(SurvivalData::new(times, events, covariates).is_err());
    }
    
    #[test]
    fn test_subset() {
        let data = create_test_data();
        let subset = data.subset(&[0, 2, 4]).unwrap();
        
        assert_eq!(subset.n_samples(), 3);
        assert_eq!(subset.times()[0], 1.0);
        assert_eq!(subset.times()[1], 3.0);
        assert_eq!(subset.times()[2], 5.0);
    }
    
    #[test]
    fn test_standardization() {
        let mut data = create_test_data();
        let (means, _stds) = data.standardize_covariates().unwrap();
        
        // Check that means are approximately zero
        for j in 0..data.n_features() {
            let col_mean = data.covariates().column(j).mean().unwrap();
            assert_relative_eq!(col_mean, 0.0, epsilon = 1e-10);
        }
        
        // Check original means and stds
        assert_relative_eq!(means[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(means[1], 6.0, epsilon = 1e-10);
    }
}