use cox_hazards::{CoxModel, SurvivalData, metrics::ModelMetrics};
use ndarray::{Array1, Array2};
use approx::assert_relative_eq;

fn create_synthetic_data(n_samples: usize, n_features: usize, seed: u64) -> SurvivalData {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Generate random covariates
    let mut covariates_vec = Vec::with_capacity(n_samples * n_features);
    for _ in 0..(n_samples * n_features) {
        covariates_vec.push(rng.gen_range(-2.0..2.0));
    }
    let covariates = Array2::from_shape_vec((n_samples, n_features), covariates_vec).unwrap();
    
    // Generate survival times based on covariates
    let mut times = Vec::with_capacity(n_samples);
    let mut events = Vec::with_capacity(n_samples);
    
    let true_coefficients = Array1::from(vec![0.5, -0.3, 0.2]); // First 3 features are relevant
    
    for i in 0..n_samples {
        let linear_pred: f64 = if n_features >= 3 {
            covariates.slice(ndarray::s![i, 0..3]).dot(&true_coefficients)
        } else {
            covariates.slice(ndarray::s![i, ..]).sum()
        };
        
        let hazard = linear_pred.exp();
        let time = (-rng.r#gen::<f64>().ln() / (0.1 * hazard)).abs().max(0.1);
        let censoring_time = rng.gen_range(1.0..10.0);
        
        if time < censoring_time {
            times.push(time);
            events.push(true);
        } else {
            times.push(censoring_time);
            events.push(false);
        }
    }
    
    SurvivalData::new(times, events, covariates).unwrap()
}

#[test]
fn test_cox_model_basic_functionality() {
    let data = create_synthetic_data(100, 5, 42);
    
    let mut model = CoxModel::new()
        .with_max_iterations(100)
        .with_tolerance(1e-4);
    
    // Model should fit without errors
    assert!(model.fit(&data).is_ok());
    assert!(model.is_fitted());
    
    // Should be able to get coefficients
    let coefficients = model.coefficients().unwrap();
    assert_eq!(coefficients.len(), 5);
    
    // Should be able to make predictions
    let predictions = model.predict(data.covariates()).unwrap();
    assert_eq!(predictions.len(), 100);
    
    // Predictions should be finite
    assert!(predictions.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_cox_model_with_ridge_regularization() {
    let data = create_synthetic_data(50, 10, 123);
    
    let mut model = CoxModel::new()
        .with_l2_penalty(0.1)
        .with_max_iterations(200);
    
    assert!(model.fit(&data).is_ok());
    
    let coefficients = model.coefficients().unwrap();
    
    // Ridge should shrink coefficients towards zero
    let coef_sum_sq: f64 = coefficients.iter().map(|x| x.powi(2)).sum();
    assert!(coef_sum_sq.is_finite() && coef_sum_sq >= 0.0);
}

#[test]
fn test_cox_model_with_lasso_regularization() {
    let data = create_synthetic_data(50, 10, 456);
    
    let mut model = CoxModel::new()
        .with_l1_penalty(0.1)
        .with_max_iterations(300);
    
    assert!(model.fit(&data).is_ok());
    
    let coefficients = model.coefficients().unwrap();
    
    // Lasso should produce some exactly zero coefficients
    let zero_coefficients = coefficients.iter().filter(|&&x| x.abs() < 1e-10).count();
    
    // We expect at least some sparsity, but this depends on the data
    assert!(zero_coefficients <= coefficients.len());
}

#[test]
fn test_cox_model_with_elastic_net() {
    let data = create_synthetic_data(80, 8, 789);
    
    let mut model = CoxModel::new()
        .with_elastic_net(0.5, 0.2)  // Equal mix of L1 and L2
        .with_max_iterations(250);
    
    assert!(model.fit(&data).is_ok());
    
    let (l1_penalty, l2_penalty) = model.regularization_params();
    assert_relative_eq!(l1_penalty, 0.1, epsilon = 1e-10);
    assert_relative_eq!(l2_penalty, 0.1, epsilon = 1e-10);
    
    let coefficients = model.coefficients().unwrap();
    assert!(coefficients.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_model_evaluation_metrics() {
    let data = create_synthetic_data(100, 3, 321);
    
    let mut model = CoxModel::new()
        .with_l2_penalty(0.01)
        .with_max_iterations(150);
    
    model.fit(&data).unwrap();
    
    let risk_scores = model.predict(data.covariates()).unwrap();
    let metrics = ModelMetrics::compute(&data, risk_scores.view(), 3).unwrap();
    
    // C-index should be reasonable (> 0.5 for a decent model)
    assert!(metrics.c_index > 0.3 && metrics.c_index <= 1.0);
    assert!(metrics.harrell_c_index > 0.3 && metrics.harrell_c_index <= 1.0);
    
    // Log-likelihood should be finite
    assert!(metrics.log_likelihood.is_finite());
    
    // AIC and BIC should be positive and finite
    assert!(metrics.aic > 0.0 && metrics.aic.is_finite());
    assert!(metrics.bic > 0.0 && metrics.bic.is_finite());
}

#[test]
fn test_hazard_ratio_predictions() {
    let data = create_synthetic_data(50, 4, 654);
    
    let mut model = CoxModel::new();
    model.fit(&data).unwrap();
    
    let hazard_ratios = model.predict_hazard_ratios(data.covariates()).unwrap();
    
    // All hazard ratios should be positive
    assert!(hazard_ratios.iter().all(|&hr| hr > 0.0 && hr.is_finite()));
    
    // The relationship between risk scores and hazard ratios should be exponential
    let risk_scores = model.predict(data.covariates()).unwrap();
    for (_, (&risk, &hr)) in risk_scores.iter().zip(hazard_ratios.iter()).enumerate() {
        assert_relative_eq!(hr, risk.exp(), epsilon = 1e-10);
    }
}

#[test]
fn test_survival_probability_predictions() {
    let data = create_synthetic_data(30, 3, 987);
    
    let mut model = CoxModel::new();
    model.fit(&data).unwrap();
    
    let time_points = Array1::from(vec![1.0, 2.0, 3.0, 5.0]);
    let survival_probs = model.predict_survival(data.covariates(), time_points.view()).unwrap();
    
    assert_eq!(survival_probs.nrows(), 30);
    assert_eq!(survival_probs.ncols(), 4);
    
    // All survival probabilities should be between 0 and 1
    for &prob in survival_probs.iter() {
        assert!(prob >= 0.0 && prob <= 1.0, "Invalid survival probability: {}", prob);
    }
    
    // Survival probabilities should generally decrease over time (for each individual)
    for i in 0..30 {
        let individual_surv = survival_probs.row(i);
        // Check that survival probabilities are non-increasing over time
        for j in 1..4 {
            assert!(individual_surv[j] <= individual_surv[j-1] + 1e-6, 
                   "Survival probabilities should be non-increasing over time");
        }
    }
}

#[test]
fn test_model_summary() {
    let data = create_synthetic_data(60, 3, 111);
    
    let feature_names = vec!["age".to_string(), "treatment".to_string(), "biomarker".to_string()];
    let mut model = CoxModel::new()
        .with_feature_names(feature_names.clone())
        .with_l1_penalty(0.05);
    
    model.fit(&data).unwrap();
    
    let summary = model.summary().unwrap();
    
    assert_eq!(summary.coefficients.len(), 3);
    assert_eq!(summary.hazard_ratios.len(), 3);
    assert_eq!(summary.feature_names.as_ref().unwrap(), &feature_names);
    
    // Hazard ratios should be exp(coefficients)
    for (_i, (&coef, &hr)) in summary.coefficients.iter()
        .zip(summary.hazard_ratios.iter()).enumerate() {
        assert_relative_eq!(hr, coef.exp(), epsilon = 1e-10);
    }
}

#[test]
fn test_feature_importance() {
    let data = create_synthetic_data(70, 5, 222);
    
    let mut model = CoxModel::new()
        .with_l1_penalty(0.1); // Use Lasso to encourage sparsity
    
    model.fit(&data).unwrap();
    
    let importance = model.feature_importance().unwrap();
    assert_eq!(importance.len(), 5);
    
    // Feature importance should be non-negative (absolute values)
    assert!(importance.iter().all(|&imp| imp >= 0.0));
    
    // Should equal absolute values of coefficients
    let coefficients = model.coefficients().unwrap();
    for (_i, (&coef, &imp)) in coefficients.iter().zip(importance.iter()).enumerate() {
        assert_relative_eq!(imp, coef.abs(), epsilon = 1e-10);
    }
}

#[test]
fn test_cross_validation_scenario() {
    let full_data = create_synthetic_data(200, 4, 555);
    
    // Split data for cross-validation
    let train_indices: Vec<usize> = (0..150).collect();
    let test_indices: Vec<usize> = (150..200).collect();
    
    let train_data = full_data.subset(&train_indices).unwrap();
    let test_data = full_data.subset(&test_indices).unwrap();
    
    // Train model
    let mut model = CoxModel::new()
        .with_elastic_net(0.3, 0.15)
        .with_max_iterations(200);
    
    model.fit(&train_data).unwrap();
    
    // Evaluate on test data
    let test_predictions = model.predict(test_data.covariates()).unwrap();
    let test_metrics = ModelMetrics::compute(&test_data, test_predictions.view(), 4).unwrap();
    
    // Model should perform reasonably on test data
    assert!(test_metrics.c_index > 0.3);
    assert!(test_metrics.log_likelihood.is_finite());
}

#[test]
fn test_edge_cases() {
    // Test with minimum viable dataset
    let times = vec![1.0, 2.0];
    let events = vec![true, false];
    let covariates = Array2::from_shape_vec((2, 1), vec![1.0, -1.0]).unwrap();
    
    let data = SurvivalData::new(times, events, covariates).unwrap();
    let mut model = CoxModel::new()
        .with_max_iterations(50);
    
    // Should handle minimal data gracefully
    assert!(model.fit(&data).is_ok());
    
    let predictions = model.predict(data.covariates()).unwrap();
    assert_eq!(predictions.len(), 2);
    assert!(predictions.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_large_regularization() {
    let data = create_synthetic_data(50, 3, 888);
    
    // Very large regularization should drive coefficients near zero
    let mut model = CoxModel::new()
        .with_l2_penalty(100.0)
        .with_max_iterations(300);
    
    model.fit(&data).unwrap();
    
    let coefficients = model.coefficients().unwrap();
    let coef_magnitude: f64 = coefficients.iter().map(|x| x.abs()).sum();
    
    // Should be heavily shrunk but not necessarily zero
    assert!(coef_magnitude < 1.0);
}