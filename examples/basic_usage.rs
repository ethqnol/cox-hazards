use cox_hazards::{CoxModel, SurvivalData, metrics::ModelMetrics};
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Cox Proportional Hazards Model - Basic Usage Example");
    println!("====================================================\n");
    
    // Create sample survival data
    let times = vec![1.2, 2.1, 3.5, 4.2, 5.8, 6.1, 7.3, 8.9, 9.2, 10.5,
                     2.3, 3.1, 4.8, 5.2, 6.9, 7.1, 8.3, 9.8, 10.1, 11.2];
    
    let events = vec![true, false, true, true, false, true, true, false, true, false,
                      true, true, false, true, true, false, true, true, false, true];
    
    // Covariates: age, treatment (0/1), biomarker_level
    let covariates = Array2::from_shape_vec((20, 3), vec![
        // age, treatment, biomarker
        65.0, 0.0, 2.3,  // Patient 1
        70.0, 1.0, 1.8,  // Patient 2
        55.0, 0.0, 3.1,  // Patient 3
        62.0, 1.0, 2.1,  // Patient 4
        68.0, 0.0, 2.8,  // Patient 5
        72.0, 1.0, 1.5,  // Patient 6
        58.0, 0.0, 3.4,  // Patient 7
        66.0, 1.0, 1.9,  // Patient 8
        71.0, 0.0, 2.6,  // Patient 9
        59.0, 1.0, 2.0,  // Patient 10
        63.0, 0.0, 2.9,  // Patient 11
        69.0, 1.0, 1.7,  // Patient 12
        57.0, 0.0, 3.2,  // Patient 13
        64.0, 1.0, 2.2,  // Patient 14
        67.0, 0.0, 2.7,  // Patient 15
        73.0, 1.0, 1.6,  // Patient 16
        61.0, 0.0, 3.0,  // Patient 17
        65.0, 1.0, 1.8,  // Patient 18
        70.0, 0.0, 2.5,  // Patient 19
        56.0, 1.0, 2.4,  // Patient 20
    ])?;
    
    // Create survival data
    let data = SurvivalData::new(times, events, covariates)?;
    
    println!("Dataset Information:");
    println!("  - Number of samples: {}", data.n_samples());
    println!("  - Number of features: {}", data.n_features());
    println!("  - Number of events: {}", data.events().iter().filter(|&&e| e).count());
    println!("  - Number of censored: {}", data.events().iter().filter(|&&e| !e).count());
    println!();
    
    // Example 1: Basic Cox regression (no regularization)
    println!("Example 1: Basic Cox Regression");
    println!("-------------------------------");
    
    let feature_names = vec![
        "Age".to_string(),
        "Treatment".to_string(), 
        "Biomarker Level".to_string()
    ];
    
    let mut basic_model = CoxModel::new()
        .with_feature_names(feature_names.clone())
        .with_max_iterations(1000)
        .with_tolerance(1e-6);
    
    basic_model.fit(&data)?;
    
    let basic_summary = basic_model.summary()?;
    basic_summary.print();
    println!();
    
    // Compute and display metrics
    let risk_scores = basic_model.predict(data.covariates())?;
    let basic_metrics = ModelMetrics::compute(&data, risk_scores.view(), 3)?;
    basic_metrics.print();
    println!("\n");
    
    // Example 2: Ridge regression (L2 regularization)
    println!("Example 2: Cox Regression with Ridge Regularization");
    println!("--------------------------------------------------");
    
    let mut ridge_model = CoxModel::new()
        .with_feature_names(feature_names.clone())
        .with_l2_penalty(0.1)
        .with_max_iterations(1000);
    
    ridge_model.fit(&data)?;
    
    let ridge_summary = ridge_model.summary()?;
    ridge_summary.print();
    println!();
    
    let ridge_risk_scores = ridge_model.predict(data.covariates())?;
    let ridge_metrics = ModelMetrics::compute(&data, ridge_risk_scores.view(), 3)?;
    ridge_metrics.print();
    println!("\n");
    
    // Example 3: Lasso regression (L1 regularization)
    println!("Example 3: Cox Regression with Lasso Regularization");
    println!("---------------------------------------------------");
    
    let mut lasso_model = CoxModel::new()
        .with_feature_names(feature_names.clone())
        .with_l1_penalty(0.1)
        .with_max_iterations(1000);
    
    lasso_model.fit(&data)?;
    
    let lasso_summary = lasso_model.summary()?;
    lasso_summary.print();
    println!();
    
    let lasso_risk_scores = lasso_model.predict(data.covariates())?;
    let lasso_metrics = ModelMetrics::compute(&data, lasso_risk_scores.view(), 3)?;
    lasso_metrics.print();
    println!("\n");
    
    // Example 4: Elastic Net regression (L1 + L2 regularization)
    println!("Example 4: Cox Regression with Elastic Net Regularization");
    println!("--------------------------------------------------------");
    
    let mut elastic_model = CoxModel::new()
        .with_feature_names(feature_names.clone())
        .with_elastic_net(0.5, 0.2)  // 50% L1, 50% L2, total penalty = 0.2
        .with_max_iterations(1000);
    
    elastic_model.fit(&data)?;
    
    let elastic_summary = elastic_model.summary()?;
    elastic_summary.print();
    println!();
    
    let elastic_risk_scores = elastic_model.predict(data.covariates())?;
    let elastic_metrics = ModelMetrics::compute(&data, elastic_risk_scores.view(), 3)?;
    elastic_metrics.print();
    println!("\n");
    
    // Example 5: Making predictions for new patients
    println!("Example 5: Predictions for New Patients");
    println!("---------------------------------------");
    
    // New patients data
    let new_patients = Array2::from_shape_vec((3, 3), vec![
        60.0, 0.0, 2.5,  // Patient A: age 60, no treatment, biomarker 2.5
        75.0, 1.0, 1.2,  // Patient B: age 75, treatment, biomarker 1.2  
        52.0, 0.0, 3.8,  // Patient C: age 52, no treatment, biomarker 3.8
    ])?;
    
    println!("Predictions using the basic Cox model:");
    let new_risk_scores = basic_model.predict(new_patients.view())?;
    let new_hazard_ratios = basic_model.predict_hazard_ratios(new_patients.view())?;
    
    for (i, (risk_score, hazard_ratio)) in new_risk_scores.iter()
        .zip(new_hazard_ratios.iter()).enumerate() {
        println!("  Patient {}: Risk Score = {:.4}, Hazard Ratio = {:.4}",
                 ('A' as u8 + i as u8) as char, risk_score, hazard_ratio);
    }
    println!();
    
    // Example 6: Survival probability predictions
    println!("Example 6: Survival Probability Predictions");
    println!("-------------------------------------------");
    
    let time_points = Array1::from(vec![1.0, 2.0, 3.0, 5.0, 10.0]);
    let survival_probs = basic_model.predict_survival(new_patients.view(), time_points.view())?;
    
    println!("Survival probabilities at different time points:");
    println!("Time:       1.0    2.0    3.0    5.0   10.0");
    for i in 0..3 {
        print!("Patient {}: ", ('A' as u8 + i as u8) as char);
        for j in 0..5 {
            print!(" {:.3}", survival_probs[[i, j]]);
        }
        println!();
    }
    println!();
    
    // Example 7: Feature importance
    println!("Example 7: Feature Importance Analysis");
    println!("-------------------------------------");
    
    let basic_importance = basic_model.feature_importance()?;
    let lasso_importance = lasso_model.feature_importance()?;
    
    println!("Feature importance (absolute coefficients):");
    println!("{:<20} {:>10} {:>10}", "Feature", "Basic", "Lasso");
    println!("{:-<40}", "");
    
    for (i, name) in feature_names.iter().enumerate() {
        println!("{:<20} {:>10.4} {:>10.4}", 
                name, basic_importance[i], lasso_importance[i]);
    }
    println!();
    
    // Example 8: Model comparison
    println!("Example 8: Model Comparison");
    println!("--------------------------");
    
    println!("{:<15} {:>10} {:>10} {:>10} {:>10}", 
             "Model", "C-index", "Log-lik", "AIC", "BIC");
    println!("{:-<55}", "");
    
    let models_metrics = [
        ("Basic", basic_metrics),
        ("Ridge", ridge_metrics),
        ("Lasso", lasso_metrics),
        ("Elastic Net", elastic_metrics),
    ];
    
    for (name, metrics) in models_metrics.iter() {
        println!("{:<15} {:>10.4} {:>10.2} {:>10.1} {:>10.1}",
                name, metrics.c_index, metrics.log_likelihood, metrics.aic, metrics.bic);
    }
    
    println!("\nInterpretation:");
    println!("- C-index: Higher is better (0.5 = random, 1.0 = perfect)");
    println!("- Log-likelihood: Higher is better");
    println!("- AIC/BIC: Lower is better (penalizes complexity)");
    
    Ok(())
}