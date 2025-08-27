use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoxError>;

#[derive(Error, Debug, Clone)]
pub enum CoxError {
    #[error("dimensions don't match: {message}")]
    InvalidDimensions { message: String },
    
    #[error("optimization blew up: {message}")]
    OptimizationFailed { message: String },
    
    #[error("model not fitted yet - call fit() first")]
    ModelNotFitted,
    
    #[error("bad parameter: {parameter} = {value}")]
    InvalidParameter { parameter: String, value: String },
    
    #[error("numerical issues: {message}")]
    NumericalError { message: String },
    
    #[error("survival data is broken: {message}")]
    InvalidSurvivalData { message: String },
}

impl CoxError {
    pub fn invalid_dimensions(message: impl Into<String>) -> Self {
        Self::InvalidDimensions { message: message.into() }
    }
    
    pub fn optimization_failed(message: impl Into<String>) -> Self {
        Self::OptimizationFailed { message: message.into() }
    }
    
    pub fn invalid_parameter(parameter: impl Into<String>, value: impl Into<String>) -> Self {
        Self::InvalidParameter { 
            parameter: parameter.into(), 
            value: value.into() 
        }
    }
    
    pub fn numerical_error(message: impl Into<String>) -> Self {
        Self::NumericalError { message: message.into() }
    }
    
    pub fn invalid_survival_data(message: impl Into<String>) -> Self {
        Self::InvalidSurvivalData { message: message.into() }
    }
}