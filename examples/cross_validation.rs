use cox_hazards::{CoxModel, SurvivalData, metrics::ModelMetrics};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn generate_synthetic_dataset(n_samples: usize, n_features: usize, seed: u64) -> cox_hazards::Result<SurvivalData> {
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Generate random covariates
    let mut covariates_vec = Vec::with_capacity(n_samples * n_features);
    for _ in 0..(n_samples * n_features) {
        covariates_vec.push(rng.gen_range(-2.0..2.0));
    }
    let covariates = Array2::from_shape_vec((n_samples, n_features), covariates_vec).unwrap();
    
    // True coefficients - first 3 features are truly predictive
    let true_coefficients = Array1::from(vec![0.8, -0.5, 0.3]);
    
    let mut times = Vec::with_capacity(n_samples);
    let mut events = Vec::with_capacity(n_samples);
    
    for i in 0..n_samples {
        // Only first 3 features affect survival
        let linear_pred: f64 = if n_features >= 3 {
            covariates.slice(ndarray::s![i, 0..3]).dot(&true_coefficients)
        } else {
            covariates.row(i).dot(&true_coefficients.slice(ndarray::s![0..n_features]))
        };
        
        let baseline_hazard = 0.1;
        let hazard = baseline_hazard * linear_pred.exp();
        
        // Generate survival time from exponential distribution
        let time = -rng.r#gen::<f64>().ln() / hazard;
        
        // Generate censoring time
        let censoring_time = rng.gen_range(1.0..15.0);
        
        if time < censoring_time {
            times.push(time);
            events.push(true);
        } else {
            times.push(censoring_time);
            events.push(false);
        }
    }
    
    SurvivalData::new(times, events, covariates)
}

fn k_fold_cross_validation(
    data: &SurvivalData,
    k: usize,
    l1_penalty: f64,
    l2_penalty: f64,
) -> cox_hazards::Result<Vec<f64>> {
    let n_samples = data.n_samples();
    let fold_size = n_samples / k;
    
    let mut c_indices = Vec::new();
    
    for fold in 0..k {
        println!("  Fold {}/{}", fold + 1, k);
        
        // Create train/test splits
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { n_samples } else { (fold + 1) * fold_size };
        
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();
        
        for i in 0..n_samples {
            if i >= test_start && i < test_end {
                test_indices.push(i);
            } else {
                train_indices.push(i);
            }
        }
        
        // Create train and test datasets
        let train_data = data.subset(&train_indices)?;
        let test_data = data.subset(&test_indices)?;
        
        // Train model
        let mut model = CoxModel::new()
            .with_l1_penalty(l1_penalty)
            .with_l2_penalty(l2_penalty)
            .with_max_iterations(500)
            .with_tolerance(1e-6);
        
        model.fit(&train_data)?;
        
        // Evaluate on test set
        let test_predictions = model.predict(test_data.covariates())?;
        let test_metrics = ModelMetrics::compute(&test_data, test_predictions.view(), data.n_features())?;
        
        c_indices.push(test_metrics.c_index);
        println!("    Test C-index: {:.4}", test_metrics.c_index);
    }
    
    Ok(c_indices)
}

fn grid_search_cv(
    data: &SurvivalData,
    l1_penalties: &[f64],
    l2_penalties: &[f64],
    cv_folds: usize,
) -> cox_hazards::Result<(f64, f64, f64)> {
    let mut best_score = 0.0;
    let mut best_l1 = 0.0;
    let mut best_l2 = 0.0;
    
    println!("Grid Search Cross-Validation");
    println!("============================");
    println!("{:<8} {:<8} {:<12} {:<12}", "L1", "L2", "Mean C-idx", "Std C-idx");
    println!("{:-<40}", "");
    
    for &l1_penalty in l1_penalties {
        for &l2_penalty in l2_penalties {
            let c_indices = k_fold_cross_validation(data, cv_folds, l1_penalty, l2_penalty)?;
            
            let mean_c_index: f64 = c_indices.iter().sum::<f64>() / c_indices.len() as f64;
            let variance: f64 = c_indices.iter()
                .map(|x| (x - mean_c_index).powi(2))
                .sum::<f64>() / c_indices.len() as f64;
            let std_c_index = variance.sqrt();
            
            println!("{:<8.3} {:<8.3} {:<12.4} {:<12.4}", 
                    l1_penalty, l2_penalty, mean_c_index, std_c_index);
            
            if mean_c_index > best_score {
                best_score = mean_c_index;
                best_l1 = l1_penalty;
                best_l2 = l2_penalty;
            }
        }
    }
    
    println!("{:-<40}", "");
    println!("Best parameters: L1={:.3}, L2={:.3}, Score={:.4}", 
             best_l1, best_l2, best_score);
    println!();
    
    Ok((best_l1, best_l2, best_score))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Cox Proportional Hazards - Cross-Validation Example");
    println!("===================================================\n");
    
    // Generate synthetic dataset
    println!("Generating synthetic dataset...");
    let full_data = generate_synthetic_dataset(300, 10, 42)?;
    
    println!("Dataset created:");
    println!("  - Samples: {}", full_data.n_samples());
    println!("  - Features: {}", full_data.n_features());
    println!("  - Events: {}", full_data.events().iter().filter(|&&e| e).count());
    println!("  - Censored: {}", full_data.events().iter().filter(|&&e| !e).count());
    println!();
    
    // Split data into train/test
    let train_size = (full_data.n_samples() as f64 * 0.8) as usize;
    let train_indices: Vec<usize> = (0..train_size).collect();
    let test_indices: Vec<usize> = (train_size..full_data.n_samples()).collect();
    
    let train_data = full_data.subset(&train_indices)?;
    let test_data = full_data.subset(&test_indices)?;
    
    println!("Data split:");
    println!("  - Training set: {} samples", train_data.n_samples());
    println!("  - Test set: {} samples", test_data.n_samples());
    println!();
    
    // Define hyperparameter grid
    let l1_penalties = vec![0.0, 0.01, 0.05, 0.1, 0.2];
    let l2_penalties = vec![0.0, 0.01, 0.05, 0.1, 0.2];
    
    // Perform grid search with cross-validation
    let (best_l1, best_l2, _best_cv_score) = grid_search_cv(
        &train_data, 
        &l1_penalties, 
        &l2_penalties, 
        5
    )?;
    
    // Train final model with best hyperparameters
    println!("Training final model with best hyperparameters...");
    let mut final_model = CoxModel::new()
        .with_l1_penalty(best_l1)
        .with_l2_penalty(best_l2)
        .with_max_iterations(1000)
        .with_tolerance(1e-6);
    
    final_model.fit(&train_data)?;
    
    // Evaluate on test set
    println!("Final Model Evaluation");
    println!("=====================");
    
    let train_predictions = final_model.predict(train_data.covariates())?;
    let train_metrics = ModelMetrics::compute(&train_data, train_predictions.view(), full_data.n_features())?;
    
    let test_predictions = final_model.predict(test_data.covariates())?;
    let test_metrics = ModelMetrics::compute(&test_data, test_predictions.view(), full_data.n_features())?;
    
    println!("Training Set Performance:");
    train_metrics.print();
    println!();
    
    println!("Test Set Performance:");
    test_metrics.print();
    println!();
    
    // Compare different regularization approaches
    println!("Comparison of Different Approaches");
    println!("=================================");
    
    let approaches = vec![
        ("No Regularization", 0.0, 0.0),
        ("Ridge Only", 0.0, 0.1),
        ("Lasso Only", 0.1, 0.0),
        ("Best Elastic Net", best_l1, best_l2),
    ];
    
    println!("{:<20} {:<10} {:<10} {:<10} {:<10}", 
             "Approach", "Train C-idx", "Test C-idx", "AIC", "BIC");
    println!("{:-<60}", "");
    
    for (name, l1, l2) in approaches {
        let mut model = CoxModel::new()
            .with_l1_penalty(l1)
            .with_l2_penalty(l2)
            .with_max_iterations(1000);
        
        model.fit(&train_data)?;
        
        let train_pred = model.predict(train_data.covariates())?;
        let train_met = ModelMetrics::compute(&train_data, train_pred.view(), full_data.n_features())?;
        
        let test_pred = model.predict(test_data.covariates())?;
        let test_met = ModelMetrics::compute(&test_data, test_pred.view(), full_data.n_features())?;
        
        println!("{:<20} {:<10.4} {:<10.4} {:<10.1} {:<10.1}",
                name, train_met.c_index, test_met.c_index, test_met.aic, test_met.bic);
    }
    
    println!();
    
    // Feature selection analysis with best model
    println!("Feature Importance Analysis (Best Model)");
    println!("=======================================");
    
    let coefficients = final_model.coefficients()?;
    let importance = final_model.feature_importance()?;
    
    println!("{:<10} {:>12} {:>12} {:>10}", "Feature", "Coefficient", "Importance", "Selected");
    println!("{:-<46}", "");
    
    for i in 0..coefficients.len() {
        let selected = if importance[i] > 1e-6 { "Yes" } else { "No" };
        println!("{:<10} {:>12.6} {:>12.6} {:>10}",
                format!("X{}", i), coefficients[i], importance[i], selected);
    }
    
    println!();
    println!("Note: Features X0, X1, X2 are the true predictive features in this synthetic dataset.");
    
    // Learning curve analysis
    println!("Learning Curve Analysis");
    println!("======================");
    
    let train_sizes = vec![50, 100, 150, 200, train_data.n_samples()];
    println!("{:<12} {:<12} {:<12}", "Train Size", "Train C-idx", "Test C-idx");
    println!("{:-<36}", "");
    
    for &size in &train_sizes {
        if size > train_data.n_samples() {
            continue;
        }
        
        let subset_indices: Vec<usize> = (0..size).collect();
        let subset_data = train_data.subset(&subset_indices)?;
        
        let mut model = CoxModel::new()
            .with_l1_penalty(best_l1)
            .with_l2_penalty(best_l2)
            .with_max_iterations(1000);
        
        model.fit(&subset_data)?;
        
        let subset_pred = model.predict(subset_data.covariates())?;
        let subset_met = ModelMetrics::compute(&subset_data, subset_pred.view(), full_data.n_features())?;
        
        let test_pred = model.predict(test_data.covariates())?;
        let test_met = ModelMetrics::compute(&test_data, test_pred.view(), full_data.n_features())?;
        
        println!("{:<12} {:<12.4} {:<12.4}", size, subset_met.c_index, test_met.c_index);
    }
    
    println!("\nCross-validation completed successfully!");
    println!("Best model achieved {:.4} C-index on test set.", test_metrics.c_index);
    
    Ok(())
}