# Cox Proportional Hazards Regression

[![Crates.io](https://img.shields.io/crates/v/cox-hazards.svg)](https://crates.io/crates/cox-hazards)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yourusername/cox-hazards/blob/main/LICENSE)

A comprehensive, well-tested Rust library for Cox proportional hazards regression with elastic net regularization. This crate provides robust survival analysis capabilities with modern optimization techniques.

## Features

- **Cox Proportional Hazards Model**: Full implementation with partial likelihood optimization
- **Elastic Net Regularization**: L1 (Lasso) and L2 (Ridge) penalties for feature selection and regularization
- **Multiple Optimization Algorithms**: Newton-Raphson and coordinate descent solvers
- **Comprehensive Metrics**: C-index, Harrell's C-index, log-likelihood, AIC, BIC
- **Survival Predictions**: Risk scores, hazard ratios, and survival probabilities
- **Robust Data Handling**: Proper treatment of censored observations and tied event times
- **Parallel Computing**: Optional parallel processing with Rayon
- **Extensive Testing**: Comprehensive test suite with integration tests and benchmarks

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
cox-hazards = "0.1.0"
```

### Basic Usage

```rust
use cox_hazards::{CoxModel, SurvivalData};
use ndarray::Array2;

// Create survival data
let times = vec![1.0, 2.5, 3.2, 4.1, 5.8];
let events = vec![true, false, true, true, false]; // true = event, false = censored
let covariates = Array2::from_shape_vec((5, 2), vec![
    1.0, 0.5,  // Patient 1: age=1.0, treatment=0.5
    2.0, 1.0,  // Patient 2: age=2.0, treatment=1.0
    1.5, 0.0,  // Patient 3: age=1.5, treatment=0.0
    3.0, 1.5,  // Patient 4: age=3.0, treatment=1.5
    2.5, 0.8,  // Patient 5: age=2.5, treatment=0.8
]).unwrap();

let data = SurvivalData::new(times, events, covariates)?;

// Fit Cox model
let mut model = CoxModel::new();
model.fit(&data)?;

// Make predictions
let risk_scores = model.predict(data.covariates())?;
let hazard_ratios = model.predict_hazard_ratios(data.covariates())?;

println!("Risk scores: {:?}", risk_scores);
println!("Hazard ratios: {:?}", hazard_ratios);
```

### Regularized Models

```rust
use cox_hazards::CoxModel;

// Ridge regression (L2 regularization)
let mut ridge_model = CoxModel::new()
    .with_l2_penalty(0.1);

// Lasso regression (L1 regularization)  
let mut lasso_model = CoxModel::new()
    .with_l1_penalty(0.1);

// Elastic net (combination of L1 and L2)
let mut elastic_model = CoxModel::new()
    .with_elastic_net(0.5, 0.2);  // 50% L1, 50% L2, total penalty = 0.2

// Fit models
ridge_model.fit(&data)?;
lasso_model.fit(&data)?;
elastic_model.fit(&data)?;
```

### Model Evaluation

```rust
use cox_hazards::metrics::ModelMetrics;

// Compute comprehensive metrics
let risk_scores = model.predict(data.covariates())?;
let metrics = ModelMetrics::compute(&data, risk_scores.view(), data.n_features())?;

println!("C-index: {:.4}", metrics.c_index);
println!("Harrell's C-index: {:.4}", metrics.harrell_c_index);
println!("Log-likelihood: {:.4}", metrics.log_likelihood);
println!("AIC: {:.4}", metrics.aic);
println!("BIC: {:.4}", metrics.bic);
```

### Advanced Features

```rust
// Model with feature names and custom parameters
let feature_names = vec!["Age".to_string(), "Treatment".to_string()];
let mut advanced_model = CoxModel::new()
    .with_feature_names(feature_names)
    .with_elastic_net(0.3, 0.1)
    .with_max_iterations(1000)
    .with_tolerance(1e-6);

advanced_model.fit(&data)?;

// Get model summary
let summary = advanced_model.summary()?;
summary.print();

// Feature importance
let importance = advanced_model.feature_importance()?;
println!("Feature importance: {:?}", importance);

// Survival probability predictions
use ndarray::Array1;
let time_points = Array1::from(vec![1.0, 2.0, 3.0, 5.0]);
let survival_probs = advanced_model.predict_survival(
    data.covariates(), 
    time_points.view()
)?;
```

## API Overview

### Core Types

- **`SurvivalData`**: Container for survival times, event indicators, and covariates
- **`CoxModel`**: Main Cox regression model with configurable regularization
- **`ModelMetrics`**: Comprehensive evaluation metrics for survival models

### Key Methods

#### SurvivalData
- `new(times, events, covariates)`: Create survival dataset
- `standardize_covariates()`: Standardize feature matrix
- `subset(indices)`: Create subset of the data

#### CoxModel
- `new()`: Create new model instance
- `with_l1_penalty(penalty)`: Set Lasso regularization
- `with_l2_penalty(penalty)`: Set Ridge regularization  
- `with_elastic_net(alpha, penalty)`: Set elastic net parameters
- `fit(data)`: Train the model
- `predict(covariates)`: Get risk scores
- `predict_hazard_ratios(covariates)`: Get hazard ratios
- `predict_survival(covariates, times)`: Get survival probabilities

#### Metrics
- `concordance_index()`: Standard C-index
- `harrell_c_index()`: Harrell's C-index with tie handling
- `log_partial_likelihood()`: Model log-likelihood
- `ModelMetrics::compute()`: All metrics at once

## Examples

The `examples/` directory contains comprehensive examples:

- **`basic_usage.rs`**: Introduction to all major features
- **`cross_validation.rs`**: Cross-validation and hyperparameter tuning

Run examples with:

```bash
cargo run --example basic_usage
cargo run --example cross_validation
```

## Mathematical Background

This implementation follows the standard Cox proportional hazards model:

**Hazard Function:**
```
h(t|x) = h₀(t) × exp(β'x)
```

**Partial Likelihood:**
```
L(β) = ∏ᵢ [exp(β'xᵢ) / Σⱼ∈R(tᵢ) exp(β'xⱼ)]^δᵢ
```

**Elastic Net Penalty:**
```
P(β) = λ₁||β||₁ + λ₂||β||₂²
```

Where:
- `h₀(t)` is the baseline hazard
- `β` are the regression coefficients  
- `x` are the covariates
- `δ` are the event indicators
- `R(t)` is the risk set at time `t`
- `λ₁, λ₂` are the regularization parameters

## Performance

The library is optimized for performance with:

- Efficient sparse matrix operations using `ndarray`
- Optional parallel processing with `rayon`
- Numerically stable likelihood computations
- Memory-efficient risk set calculations

Run benchmarks with:

```bash
cargo bench
```

## Testing

Comprehensive test coverage including:

- Unit tests for all components
- Integration tests with synthetic data  
- Property-based testing with `proptest`
- Numerical accuracy verification

Run tests with:

```bash
cargo test
```

## Features

- **`default`**: Includes parallel processing
- **`parallel`**: Enable parallel computations with Rayon

## Requirements

- Rust 1.70.0 or later
- Compatible with `no_std` environments (without `std` feature)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{cox_hazards_rust,
  title = {cox-hazards: Cox Proportional Hazards Regression in Rust},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/cox-hazards},
}
```

## References

1. Cox, D. R. (1972). Regression models and life‐tables. *Journal of the Royal Statistical Society*, 34(2), 187-202.
2. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society*, 67(2), 301-320.
3. Harrell Jr, F. E., et al. (1996). Multivariable prognostic models. *Statistics in Medicine*, 15(4), 361-387.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with the excellent Rust scientific computing ecosystem
- Inspired by R's `survival` package and Python's `lifelines`
- Uses `ndarray` for efficient numerical computing
- Optimization powered by `argmin` framework