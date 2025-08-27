use ndarray::{Array1, Array2};
use crate::{
    data::SurvivalData,
    error::{CoxError, Result},
};


/// Configuration for Cox model optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub l1_penalty: f64,
    pub l2_penalty: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            l1_penalty: 0.0,
            l2_penalty: 0.0,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

/// Cox proportional hazards optimizer with elastic net regularization
pub struct CoxOptimizer {
    config: OptimizationConfig,
}

impl CoxOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }
    
    /// Optimize Cox model using Newton-Raphson with elastic net regularization
    pub fn optimize(&self, data: &SurvivalData) -> Result<Array1<f64>> {
        let n_features = data.n_features();
        let mut beta = Array1::zeros(n_features);
        
        // Use coordinate descent for elastic net regularization
        if self.config.l1_penalty > 0.0 {
            self.coordinate_descent_optimize(data, &mut beta)?;
        } else {
            // Pure Ridge regression - use Newton-Raphson
            self.newton_raphson_optimize(data, &mut beta)?;
        }
        
        Ok(beta)
    }
    
    /// Newton-Raphson optimization (for Ridge regression)
    fn newton_raphson_optimize(&self, data: &SurvivalData, beta: &mut Array1<f64>) -> Result<()> {
        let mut prev_loglik = f64::NEG_INFINITY;
        
        for iteration in 0..self.config.max_iterations {
            let (loglik, gradient, hessian) = self.compute_likelihood_derivatives(data, beta)?;
            
            // Add Ridge penalty
            let penalized_loglik = loglik - 0.5 * self.config.l2_penalty * beta.dot(beta);
            
            // Check for convergence
            if (penalized_loglik - prev_loglik).abs() < self.config.tolerance {
                break;
            }
            
            if iteration == self.config.max_iterations - 1 {
                return Err(CoxError::optimization_failed(
                    "Newton-Raphson failed to converge"
                ));
            }
            
            // Add Ridge penalty to gradient and Hessian
            let penalized_gradient = &gradient - self.config.l2_penalty * &*beta;
            let mut penalized_hessian = hessian.clone();
            for i in 0..beta.len() {
                penalized_hessian[[i, i]] -= self.config.l2_penalty;
            }
            
            // Newton-Raphson step
            match self.solve_linear_system(&penalized_hessian, &penalized_gradient) {
                Ok(step) => {
                    *beta = beta.clone() - step;
                }
                Err(_) => {
                    // Fall back to gradient descent
                    let step_size = 0.01;
                    *beta = beta.clone() + step_size * &penalized_gradient;
                }
            }
            
            prev_loglik = penalized_loglik;
        }
        
        Ok(())
    }
    
    /// Coordinate descent optimization (for elastic net)
    fn coordinate_descent_optimize(&self, data: &SurvivalData, beta: &mut Array1<f64>) -> Result<()> {
        let n_features = data.n_features();
        
        for iteration in 0..self.config.max_iterations {
            let mut converged = true;
            let _beta_old = beta.clone();
            
            for j in 0..n_features {
                let beta_old_j = beta[j];
                
                // Compute partial residuals
                let partial_gradient = self.compute_partial_gradient(data, beta, j)?;
                let partial_hessian = self.compute_partial_hessian(data, beta, j)?;
                
                // Coordinate-wise update with soft thresholding
                let raw_update = beta[j] + partial_gradient / partial_hessian.abs().max(1e-8);
                beta[j] = self.soft_threshold(raw_update, self.config.l1_penalty / partial_hessian.abs().max(1e-8));
                
                // Add Ridge penalty
                if self.config.l2_penalty > 0.0 {
                    beta[j] /= 1.0 + self.config.l2_penalty / partial_hessian.abs().max(1e-8);
                }
                
                if (beta[j] - beta_old_j).abs() > self.config.tolerance {
                    converged = false;
                }
            }
            
            if converged {
                break;
            }
            
            if iteration == self.config.max_iterations - 1 {
                return Err(CoxError::optimization_failed(
                    "Coordinate descent failed to converge"
                ));
            }
        }
        
        Ok(())
    }
    
    /// Soft thresholding operator for L1 regularization
    fn soft_threshold(&self, x: f64, lambda: f64) -> f64 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }
    
    /// Compute partial gradient for coordinate j
    fn compute_partial_gradient(&self, data: &SurvivalData, beta: &Array1<f64>, j: usize) -> Result<f64> {
        let mut gradient = 0.0;
        let event_times = data.event_times();
        
        for &event_time in &event_times {
            let events_at_time: Vec<usize> = (0..data.n_samples())
                .filter(|&i| data.times()[i] == event_time && data.events()[i])
                .collect();
            
            if events_at_time.is_empty() {
                continue;
            }
            
            let risk_set: Vec<usize> = (0..data.n_samples())
                .filter(|&i| data.times()[i] >= event_time)
                .collect();
            
            if risk_set.is_empty() {
                continue;
            }
            
            // Calculate risk set sum
            let mut risk_sum = 0.0;
            let mut weighted_covariate_sum = 0.0;
            
            for &i in &risk_set {
                let linear_pred = data.covariates().row(i).dot(beta);
                let exp_pred = linear_pred.exp();
                risk_sum += exp_pred;
                weighted_covariate_sum += data.covariates()[[i, j]] * exp_pred;
            }
            
            if risk_sum <= 0.0 {
                return Err(CoxError::numerical_error("Risk set sum is non-positive"));
            }
            
            // Add contribution from each event
            for &event_idx in &events_at_time {
                gradient += data.covariates()[[event_idx, j]] - weighted_covariate_sum / risk_sum;
            }
        }
        
        Ok(gradient)
    }
    
    /// Compute partial Hessian for coordinate j
    fn compute_partial_hessian(&self, data: &SurvivalData, beta: &Array1<f64>, j: usize) -> Result<f64> {
        let mut hessian = 0.0;
        let event_times = data.event_times();
        
        for &event_time in &event_times {
            let events_at_time: Vec<usize> = (0..data.n_samples())
                .filter(|&i| data.times()[i] == event_time && data.events()[i])
                .collect();
            
            if events_at_time.is_empty() {
                continue;
            }
            
            let risk_set: Vec<usize> = (0..data.n_samples())
                .filter(|&i| data.times()[i] >= event_time)
                .collect();
            
            if risk_set.is_empty() {
                continue;
            }
            
            let mut risk_sum = 0.0;
            let mut weighted_covariate_sum = 0.0;
            let mut weighted_covariate_squared_sum = 0.0;
            
            for &i in &risk_set {
                let linear_pred = data.covariates().row(i).dot(beta);
                let exp_pred = linear_pred.exp();
                let covariate_j = data.covariates()[[i, j]];
                
                risk_sum += exp_pred;
                weighted_covariate_sum += covariate_j * exp_pred;
                weighted_covariate_squared_sum += covariate_j * covariate_j * exp_pred;
            }
            
            if risk_sum <= 0.0 {
                return Err(CoxError::numerical_error("Risk set sum is non-positive"));
            }
            
            // Second derivative calculation
            let first_moment = weighted_covariate_sum / risk_sum;
            let second_moment = weighted_covariate_squared_sum / risk_sum;
            
            hessian -= events_at_time.len() as f64 * (second_moment - first_moment * first_moment);
        }
        
        Ok(hessian)
    }
    
    /// Compute log partial likelihood and its derivatives
    fn compute_likelihood_derivatives(
        &self,
        data: &SurvivalData,
        beta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let n_features = data.n_features();
        let mut loglik = 0.0;
        let mut gradient = Array1::zeros(n_features);
        let mut hessian = Array2::zeros((n_features, n_features));
        
        let event_times = data.event_times();
        
        for &event_time in &event_times {
            let events_at_time: Vec<usize> = (0..data.n_samples())
                .filter(|&i| data.times()[i] == event_time && data.events()[i])
                .collect();
            
            if events_at_time.is_empty() {
                continue;
            }
            
            let risk_set: Vec<usize> = (0..data.n_samples())
                .filter(|&i| data.times()[i] >= event_time)
                .collect();
            
            if risk_set.is_empty() {
                continue;
            }
            
            // Compute risk set statistics
            let (log_sum, weighted_mean, weighted_variance) = self.compute_risk_set_statistics(data, beta, &risk_set)?;
            
            // Update likelihood and derivatives for each event
            for &event_idx in &events_at_time {
                let event_linear_pred = data.covariates().row(event_idx).dot(beta);
                loglik += event_linear_pred - log_sum;
                
                let event_covariates = data.covariates().row(event_idx).to_owned();
                gradient += &(&event_covariates - &weighted_mean);
                
                // Hessian update
                hessian -= &weighted_variance;
            }
        }
        
        Ok((loglik, gradient, hessian))
    }
    
    
    /// Compute statistics for a risk set
    fn compute_risk_set_statistics(
        &self,
        data: &SurvivalData,
        beta: &Array1<f64>,
        risk_set: &[usize],
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let n_features = data.n_features();
        let mut risk_sum = 0.0;
        let mut weighted_covariate_sum = Array1::zeros(n_features);
        let mut weighted_covariate_outer_sum = Array2::zeros((n_features, n_features));
        
        for &i in risk_set {
            let linear_pred = data.covariates().row(i).dot(beta);
            let exp_pred = linear_pred.exp();
            
            if !exp_pred.is_finite() || exp_pred <= 0.0 {
                return Err(CoxError::numerical_error(
                    format!("Invalid exponential prediction: {}", exp_pred)
                ));
            }
            
            risk_sum += exp_pred;
            let covariates_i = data.covariates().row(i).to_owned();
            weighted_covariate_sum += &(exp_pred * &covariates_i);
            
            // Outer product for Hessian
            for j in 0..n_features {
                for k in 0..n_features {
                    weighted_covariate_outer_sum[[j, k]] += 
                        exp_pred * covariates_i[j] * covariates_i[k];
                }
            }
        }
        
        if risk_sum <= 0.0 {
            return Err(CoxError::numerical_error("Risk set sum is non-positive"));
        }
        
        let log_sum = risk_sum.ln();
        let weighted_mean = &weighted_covariate_sum / risk_sum;
        
        // Compute variance matrix
        let mut weighted_variance = weighted_covariate_outer_sum / risk_sum;
        for i in 0..n_features {
            for j in 0..n_features {
                weighted_variance[[i, j]] -= weighted_mean[i] * weighted_mean[j];
            }
        }
        
        Ok((log_sum, weighted_mean, weighted_variance))
    }
    
    /// Solve linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        // Simple LU decomposition approach
        // In practice, you might want to use a more robust solver
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return Err(CoxError::invalid_dimensions("Matrix dimensions mismatch"));
        }
        
        let mut a_copy = a.clone();
        let mut b_copy = b.clone();
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if a_copy[[k, i]].abs() > a_copy[[max_row, i]].abs() {
                    max_row = k;
                }
            }
            
            if a_copy[[max_row, i]].abs() < 1e-12 {
                return Err(CoxError::numerical_error("Matrix is singular"));
            }
            
            // Swap rows
            if max_row != i {
                for j in 0..n {
                    let temp = a_copy[[i, j]];
                    a_copy[[i, j]] = a_copy[[max_row, j]];
                    a_copy[[max_row, j]] = temp;
                }
                let temp = b_copy[i];
                b_copy[i] = b_copy[max_row];
                b_copy[max_row] = temp;
            }
            
            // Eliminate
            for k in i + 1..n {
                let factor = a_copy[[k, i]] / a_copy[[i, i]];
                for j in i..n {
                    a_copy[[k, j]] -= factor * a_copy[[i, j]];
                }
                b_copy[k] -= factor * b_copy[i];
            }
        }
        
        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = b_copy[i];
            for j in i + 1..n {
                x[i] -= a_copy[[i, j]] * x[j];
            }
            x[i] /= a_copy[[i, i]];
        }
        
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use approx::assert_relative_eq;
    
    fn create_test_data() -> SurvivalData {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, true, true, true];
        let covariates = Array2::from_shape_vec((5, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            -1.0, 0.0,
            0.0, -1.0,
        ]).unwrap();
        
        SurvivalData::new(times, events, covariates).unwrap()
    }
    
    #[test]
    fn test_optimizer_creation() {
        let config = OptimizationConfig::default();
        let optimizer = CoxOptimizer::new(config.clone());
        assert_eq!(optimizer.config.l1_penalty, config.l1_penalty);
        assert_eq!(optimizer.config.l2_penalty, config.l2_penalty);
    }
    
    #[test]
    fn test_soft_threshold() {
        let config = OptimizationConfig::default();
        let optimizer = CoxOptimizer::new(config);
        
        assert_relative_eq!(optimizer.soft_threshold(2.0, 1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(optimizer.soft_threshold(-2.0, 1.0), -1.0, epsilon = 1e-10);
        assert_relative_eq!(optimizer.soft_threshold(0.5, 1.0), 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_optimization_no_regularization() {
        let data = create_test_data();
        let config = OptimizationConfig::default();
        let optimizer = CoxOptimizer::new(config);
        
        let result = optimizer.optimize(&data);
        assert!(result.is_ok());
        
        let beta = result.unwrap();
        assert_eq!(beta.len(), 2);
    }
    
    #[test]
    fn test_optimization_with_ridge() {
        let data = create_test_data();
        let config = OptimizationConfig {
            l1_penalty: 0.0,
            l2_penalty: 0.1,
            max_iterations: 100,
            tolerance: 1e-6,
        };
        let optimizer = CoxOptimizer::new(config);
        
        let result = optimizer.optimize(&data);
        assert!(result.is_ok());
        
        let beta = result.unwrap();
        assert_eq!(beta.len(), 2);
    }
    
    #[test]
    fn test_optimization_with_lasso() {
        let data = create_test_data();
        let config = OptimizationConfig {
            l1_penalty: 0.1,
            l2_penalty: 0.0,
            max_iterations: 100,
            tolerance: 1e-6,
        };
        let optimizer = CoxOptimizer::new(config);
        
        let result = optimizer.optimize(&data);
        assert!(result.is_ok());
        
        let beta = result.unwrap();
        assert_eq!(beta.len(), 2);
    }
    
    #[test]
    fn test_optimization_with_elastic_net() {
        let data = create_test_data();
        let config = OptimizationConfig {
            l1_penalty: 0.05,
            l2_penalty: 0.05,
            max_iterations: 100,
            tolerance: 1e-6,
        };
        let optimizer = CoxOptimizer::new(config);
        
        let result = optimizer.optimize(&data);
        assert!(result.is_ok());
        
        let beta = result.unwrap();
        assert_eq!(beta.len(), 2);
    }
}