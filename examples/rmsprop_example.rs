use cox_hazards::{CoxModel, SurvivalData, optimization::OptimizerType};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Cox Hazards with RMSprop Optimizer Example");
    println!("==========================================\n");
    
    // Create sample survival data
    let times = vec![1.2, 2.1, 3.5, 4.2, 5.8, 6.1, 7.3, 8.9, 9.2, 10.5];
    let events = vec![true, false, true, true, false, true, true, false, true, false];
    
    // Covariates: age, treatment, biomarker
    let covariates = Array2::from_shape_vec((10, 3), vec![
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
    ])?;
    
    let data = SurvivalData::new(times, events, covariates)?;
    
    println!("Dataset: {} samples, {} features", data.n_samples(), data.n_features());
    println!();

    // Example 1: Basic RMSprop
    println!("Example 1: Basic RMSprop");
    println!("------------------------");
    
    let mut rmsprop_model = CoxModel::new()
        .with_optimizer(OptimizerType::RMSprop)
        .with_learning_rate(0.1)
        .with_adam_params(0.9, 0.9)  // beta1 not used in RMSprop, beta2 = decay rate
        .with_epsilon(1e-8)
        .with_max_iterations(500);
    
    rmsprop_model.fit(&data)?;
    let rmsprop_summary = rmsprop_model.summary()?;
    rmsprop_summary.print();
    println!();
    
    // Example 2: RMSprop with regularization
    println!("Example 2: RMSprop with L2 Regularization");
    println!("-----------------------------------------");
    
    let mut rmsprop_l2_model = CoxModel::new()
        .with_optimizer(OptimizerType::RMSprop)
        .with_l2_penalty(0.1)
        .with_learning_rate(0.05)
        .with_adam_params(0.9, 0.9)  // beta2 = 0.9 for RMSprop decay
        .with_max_iterations(800);
    
    rmsprop_l2_model.fit(&data)?;
    let rmsprop_l2_summary = rmsprop_l2_model.summary()?;
    rmsprop_l2_summary.print();
    println!();
    
    // Example 3: Comparison with Adam
    println!("Example 3: Comparison with Adam");
    println!("-------------------------------");
    
    let mut adam_model = CoxModel::new()
        .with_optimizer(OptimizerType::Adam)
        .with_learning_rate(0.01)
        .with_adam_params(0.9, 0.999)  // standard Adam parameters
        .with_max_iterations(500);
    
    adam_model.fit(&data)?;
    let adam_summary = adam_model.summary()?;
    
    println!("Adam coefficients:");
    adam_summary.print();
    println!();
    
    println!("RMSprop vs Adam coefficients comparison:");
    println!("{:<15} {:>10} {:>10}", "Feature", "RMSprop", "Adam");
    println!("{:-<35}", "");
    
    let rmsprop_coeffs = rmsprop_model.coefficients().unwrap();
    let adam_coeffs = adam_model.coefficients().unwrap();
    
    for i in 0..rmsprop_coeffs.len() {
        println!("{:<15} {:>10.4} {:>10.4}", 
                 format!("Feature {}", i + 1), 
                 rmsprop_coeffs[i], 
                 adam_coeffs[i]);
    }
    println!();
    
    println!("Key differences between RMSprop and Adam:");
    println!("- RMSprop: Uses only second moment (moving average of squared gradients)");
    println!("- Adam: Uses both first moment (momentum) and second moment");
    println!("- RMSprop can be more stable in some cases, Adam often converges faster");
    println!("- Both are good for dealing with sparse gradients and noisy data");
    
    Ok(())
}