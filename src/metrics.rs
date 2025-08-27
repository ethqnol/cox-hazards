use ndarray::{Array1, ArrayView1};
use crate::{
    data::SurvivalData,
    error::{CoxError, Result},
};


/// concordance index - how often do higher risk scores = shorter survival? 
pub fn concordance_index(
    risk_scores: ArrayView1<f64>,
    times: ArrayView1<f64>,
    events: &[bool],
) -> Result<f64> {
    if risk_scores.len() != times.len() || times.len() != events.len() {
        return Err(CoxError::invalid_dimensions(
            "risk scores, times, and events must have same length"
        ));
    }
    
    let n = risk_scores.len();
    if n < 2 {
        return Err(CoxError::invalid_dimensions(
            "need at least 2 samples for concordance"
        ));
    }
    
    let mut concordant = 0u64;
    let mut comparable = 0u64;
    
    for i in 0..n {
        if !events[i] {
            continue; // skip censored obs as event times
        }
        
        for j in 0..n {
            if i == j {
                continue;
            }
            
            // j is comparable to i if j survived longer (event or censored)
            if times[j] > times[i] || (!events[j] && times[j] >= times[i]) {
                comparable += 1;
                
                // concordant if higher risk = shorter survival
                if risk_scores[i] > risk_scores[j] {
                    concordant += 1;
                } else if risk_scores[i] == risk_scores[j] {
                    concordant += 1; // ties get counted as 0.5 later
                }
            }
        }
    }
    
    if comparable == 0 {
        return Err(CoxError::numerical_error(
            "no comparable pairs for concordance calc"
        ));
    }
    
    Ok(concordant as f64 / comparable as f64)
}

/// Harrell's C-index with tie handling
pub fn harrell_c_index(
    risk_scores: ArrayView1<f64>,
    times: ArrayView1<f64>,
    events: &[bool],
) -> Result<f64> {
    let n = risk_scores.len();
    if n != times.len() || n != events.len() {
        return Err(CoxError::invalid_dimensions(
            "All arrays must have same length"
        ));
    }
    
    let mut concordant = 0.0;
    let mut discordant = 0.0;
    let mut tied_risk = 0.0;
    
    for i in 0..n {
        if !events[i] {
            continue;
        }
        
        for j in 0..n {
            if i == j {
                continue;
            }
            
            if times[j] > times[i] || (!events[j] && times[j] >= times[i]) {
                if risk_scores[i] > risk_scores[j] {
                    concordant += 1.0;
                } else if risk_scores[i] < risk_scores[j] {
                    discordant += 1.0;
                } else {
                    tied_risk += 1.0;
                }
            }
        }
    }
    
    let total_pairs = concordant + discordant + tied_risk;
    if total_pairs == 0.0 {
        return Err(CoxError::numerical_error(
            "No valid pairs for C-index calculation"
        ));
    }
    
    // Harrell's C-index: (concordant + 0.5 * tied) / total
    Ok((concordant + 0.5 * tied_risk) / total_pairs)
}

/// Uno's C-index (time-dependent)
pub fn uno_c_index(
    risk_scores: ArrayView1<f64>,
    times: ArrayView1<f64>,
    events: &[bool],
    tau: Option<f64>,
) -> Result<f64> {
    let n = risk_scores.len();
    if n != times.len() || n != events.len() {
        return Err(CoxError::invalid_dimensions("All arrays must have same length"));
    }
    
    let max_time = tau.unwrap_or_else(|| {
        times.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    });
    
    // Compute Kaplan-Meier weights for censoring
    let km_weights = compute_kaplan_meier_weights(times, events, max_time)?;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..n {
        if !events[i] || times[i] > max_time {
            continue;
        }
        
        for j in 0..n {
            if i == j || times[j] <= times[i] {
                continue;
            }
            
            let weight = km_weights[i] * km_weights[j];
            denominator += weight;
            
            if risk_scores[i] > risk_scores[j] {
                numerator += weight;
            } else if risk_scores[i] == risk_scores[j] {
                numerator += 0.5 * weight;
            }
        }
    }
    
    if denominator == 0.0 {
        return Err(CoxError::numerical_error("No valid pairs for Uno's C-index"));
    }
    
    Ok(numerator / denominator)
}

/// Compute Kaplan-Meier survival weights
fn compute_kaplan_meier_weights(
    times: ArrayView1<f64>,
    events: &[bool],
    max_time: f64,
) -> Result<Array1<f64>> {
    let n = times.len();
    let mut weights = Array1::ones(n);
    
    // Get unique event times
    let mut event_times: Vec<f64> = times
        .iter()
        .zip(events.iter())
        .filter_map(|(&t, &e)| if e && t <= max_time { Some(t) } else { None })
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    event_times.dedup();
    
    let mut survival_prob = 1.0;
    
    for &event_time in &event_times {
        let at_risk = times.iter().filter(|&&t| t >= event_time).count();
        let events_count = times
            .iter()
            .zip(events.iter())
            .filter(|&(&t, &e)| t == event_time && e)
            .count();
        
        if at_risk > 0 {
            survival_prob *= 1.0 - (events_count as f64 / at_risk as f64);
            
            // Update weights for censored observations at this time
            for i in 0..n {
                if times[i] == event_time && !events[i] {
                    weights[i] = survival_prob;
                }
            }
        }
    }
    
    Ok(weights)
}

/// Integrated Brier Score for survival models
pub fn integrated_brier_score(
    survival_probs: &Array1<Array1<f64>>, // Survival probabilities for each sample at each time
    evaluation_times: ArrayView1<f64>,
    actual_times: ArrayView1<f64>,
    events: &[bool],
) -> Result<f64> {
    let n_samples = actual_times.len();
    let n_times = evaluation_times.len();
    
    if survival_probs.len() != n_samples {
        return Err(CoxError::invalid_dimensions(
            "Number of survival probability arrays must match number of samples"
        ));
    }
    
    let mut total_brier = 0.0;
    
    for (t_idx, &eval_time) in evaluation_times.iter().enumerate() {
        let mut brier_at_time = 0.0;
        
        for i in 0..n_samples {
            if survival_probs[i].len() != n_times {
                return Err(CoxError::invalid_dimensions(
                    "All survival probability arrays must have same length as evaluation times"
                ));
            }
            
            let predicted_surv = survival_probs[i][t_idx];
            
            // Actual survival status at evaluation time
            let actual_surv = if actual_times[i] > eval_time || 
                             (!events[i] && actual_times[i] >= eval_time) { 1.0 } else { 0.0 };
            
            brier_at_time += (predicted_surv - actual_surv).powi(2);
        }
        
        total_brier += brier_at_time / n_samples as f64;
    }
    
    Ok(total_brier / n_times as f64)
}

/// Log-likelihood for Cox model evaluation
pub fn log_partial_likelihood(
    data: &SurvivalData,
    risk_scores: ArrayView1<f64>,
) -> Result<f64> {
    if risk_scores.len() != data.n_samples() {
        return Err(CoxError::invalid_dimensions(
            "Risk scores length must match number of samples"
        ));
    }
    
    let mut loglik = 0.0;
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
        
        // Calculate log of sum of exponentials (numerically stable)
        let max_risk = risk_set.iter().map(|&i| risk_scores[i]).fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp = max_risk + 
            risk_set.iter()
                .map(|&i| (risk_scores[i] - max_risk).exp())
                .sum::<f64>()
                .ln();
        
        // Add contribution from each event
        for &event_idx in &events_at_time {
            loglik += risk_scores[event_idx] - log_sum_exp;
        }
    }
    
    Ok(loglik)
}

/// AIC (Akaike Information Criterion) for Cox model
pub fn aic(log_likelihood: f64, n_parameters: usize) -> f64 {
    2.0 * n_parameters as f64 - 2.0 * log_likelihood
}

/// BIC (Bayesian Information Criterion) for Cox model  
pub fn bic(log_likelihood: f64, n_parameters: usize, n_samples: usize) -> f64 {
    (n_parameters as f64) * (n_samples as f64).ln() - 2.0 * log_likelihood
}

/// Comprehensive model evaluation metrics
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub c_index: f64,
    pub harrell_c_index: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
}

impl ModelMetrics {
    /// Compute all metrics for a fitted Cox model
    pub fn compute(
        data: &SurvivalData,
        risk_scores: ArrayView1<f64>,
        n_parameters: usize,
    ) -> Result<Self> {
        let c_index = concordance_index(risk_scores, data.times(), data.events())?;
        let harrell_c_index = harrell_c_index(risk_scores, data.times(), data.events())?;
        let log_likelihood = log_partial_likelihood(data, risk_scores)?;
        let aic_score = aic(log_likelihood, n_parameters);
        let bic_score = bic(log_likelihood, n_parameters, data.n_samples());
        
        Ok(Self {
            c_index,
            harrell_c_index,
            log_likelihood,
            aic: aic_score,
            bic: bic_score,
        })
    }
    
    /// Print metrics summary
    pub fn print(&self) {
        println!("Model Evaluation Metrics");
        println!("========================");
        println!("C-index:             {:.6}", self.c_index);
        println!("Harrell's C-index:   {:.6}", self.harrell_c_index);
        println!("Log-likelihood:      {:.6}", self.log_likelihood);
        println!("AIC:                 {:.6}", self.aic);
        println!("BIC:                 {:.6}", self.bic);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use crate::data::SurvivalData;
    use approx::assert_relative_eq;
    
    fn create_test_data() -> (SurvivalData, Array1<f64>) {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, false, true, true, false];
        let covariates = Array2::from_shape_vec((5, 2), vec![
            1.0, 2.0,
            0.0, 1.0,
            1.0, 0.0,
            -1.0, 1.0,
            0.0, -1.0,
        ]).unwrap();
        
        let data = SurvivalData::new(times, events, covariates).unwrap();
        let risk_scores = Array1::from(vec![0.5, -0.2, 0.8, -0.1, -0.5]);
        
        (data, risk_scores)
    }
    
    #[test]
    fn test_concordance_index() {
        let (data, risk_scores) = create_test_data();
        let c_index = concordance_index(
            risk_scores.view(),
            data.times(),
            data.events(),
        ).unwrap();
        
        assert!(c_index >= 0.0 && c_index <= 1.0);
    }
    
    #[test]
    fn test_harrell_c_index() {
        let (data, risk_scores) = create_test_data();
        let harrell_c = harrell_c_index(
            risk_scores.view(),
            data.times(),
            data.events(),
        ).unwrap();
        
        assert!(harrell_c >= 0.0 && harrell_c <= 1.0);
    }
    
    #[test]
    fn test_log_partial_likelihood() {
        let (data, risk_scores) = create_test_data();
        let loglik = log_partial_likelihood(&data, risk_scores.view()).unwrap();
        
        // Log-likelihood should be negative (or zero at maximum)
        assert!(loglik.is_finite());
    }
    
    #[test]
    fn test_aic_bic() {
        let log_likelihood = -10.0;
        let n_parameters = 3;
        let n_samples = 100;
        
        let aic_score = aic(log_likelihood, n_parameters);
        let bic_score = bic(log_likelihood, n_parameters, n_samples);
        
        assert!(aic_score > 0.0);
        assert!(bic_score > 0.0);
        assert!(bic_score > aic_score); // BIC typically higher for reasonable sample sizes
    }
    
    #[test]
    fn test_model_metrics() {
        let (data, risk_scores) = create_test_data();
        let metrics = ModelMetrics::compute(&data, risk_scores.view(), 2).unwrap();
        
        assert!(metrics.c_index >= 0.0 && metrics.c_index <= 1.0);
        assert!(metrics.harrell_c_index >= 0.0 && metrics.harrell_c_index <= 1.0);
        assert!(metrics.log_likelihood.is_finite());
        assert!(metrics.aic > 0.0);
        assert!(metrics.bic > 0.0);
    }
    
    #[test]
    fn test_perfect_concordance() {
        let times = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let events = vec![true, true, true, true];
        let risk_scores = Array1::from(vec![4.0, 3.0, 2.0, 1.0]); // Perfectly anti-correlated with time
        
        let c_index = concordance_index(risk_scores.view(), times.view(), &events).unwrap();
        assert_relative_eq!(c_index, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_dimension_mismatch_error() {
        let risk_scores = Array1::from(vec![1.0, 2.0]);
        let times = Array1::from(vec![1.0, 2.0, 3.0]); // Different length
        let events = vec![true, false];
        
        assert!(concordance_index(risk_scores.view(), times.view(), &events).is_err());
    }
}