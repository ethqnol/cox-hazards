//! # cox hazards regression
//! 
//! cox proportional hazards w/ elastic net regularization - survival analysis made easy
//! 
//! ## what you get
//! 
//! - standard cox regression 
//! - elastic net (ridge + lasso) for regularization
//! - multiple solvers that actually work
//! - all the survival metrics you need
//! - parallel processing when you want it
//! 
//! ## quick start
//! 
//! ```rust
//! use cox_hazards::{CoxModel, SurvivalData};
//! use ndarray::Array2;
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // setup some survival data
//! let times = vec![1.0, 2.5, 3.2, 4.1];
//! let events = vec![true, false, true, true]; // true = died, false = censored
//! let covariates = Array2::from_shape_vec((4, 2), vec![
//!     1.0, 0.5,  // patient features
//!     2.0, 1.0, 
//!     1.5, 0.0,
//!     3.0, 1.5,
//! ])?;
//! let data = SurvivalData::new(times, events, covariates)?;
//! 
//! // fit w/ some regularization 
//! let mut model = CoxModel::new()
//!     .with_l1_penalty(0.1)    // lasso
//!     .with_l2_penalty(0.1);   // ridge
//! 
//! model.fit(&data)?;
//! 
//! // get risk scores
//! let risk_scores = model.predict(data.covariates())?;
//! # Ok(())
//! # }
//! ```

pub mod data;
pub mod model;
pub mod optimization;
pub mod metrics;
pub mod error;

pub use data::SurvivalData;
pub use model::CoxModel;
pub use error::{CoxError, Result};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_basic_functionality() {
        let n_samples = 100;
        let n_features = 5;
        
        let times = vec![1.0; n_samples];
        let events = vec![true; n_samples];
        let covariates = Array2::zeros((n_samples, n_features));
        
        let data = SurvivalData::new(times, events, covariates).unwrap();
        assert_eq!(data.n_samples(), n_samples);
        assert_eq!(data.n_features(), n_features);
    }
}