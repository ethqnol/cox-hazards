use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cox_hazards::{CoxModel, SurvivalData};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn generate_synthetic_data(n_samples: usize, n_features: usize) -> SurvivalData {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Generate random covariates
    let mut covariates_vec = Vec::with_capacity(n_samples * n_features);
    for _ in 0..(n_samples * n_features) {
        covariates_vec.push(rng.gen_range(-2.0..2.0));
    }
    let covariates = Array2::from_shape_vec((n_samples, n_features), covariates_vec).unwrap();
    
    // Generate survival times
    let mut times = Vec::with_capacity(n_samples);
    let mut events = Vec::with_capacity(n_samples);
    
    let true_coefficients = Array1::from(vec![0.5, -0.3, 0.2]);
    
    for i in 0..n_samples {
        let n_coef = n_features.min(3);
        let linear_pred: f64 = covariates.row(i).slice(ndarray::s![0..n_coef])
            .dot(&true_coefficients.slice(ndarray::s![0..n_coef]));
        
        let hazard = linear_pred.exp();
        let time = -(-rng.r#gen::<f64>().ln() / (0.1 * hazard)).max(0.1);
        let censoring_time = rng.gen_range(1.0..8.0);
        
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

fn benchmark_cox_fitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("cox_fitting");
    
    for &n_samples in [50, 100, 200, 500].iter() {
        for &n_features in [5, 10, 20].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
                &(n_samples, n_features),
                |b, &(n_samples, n_features)| {
                    let data = generate_synthetic_data(n_samples, n_features);
                    b.iter(|| {
                        let mut model = CoxModel::new()
                            .with_max_iterations(100)
                            .with_tolerance(1e-4);
                        model.fit(black_box(&data)).unwrap();
                    });
                },
            );
        }
    }
    group.finish();
}

fn benchmark_elastic_net_fitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("elastic_net_fitting");
    
    let data = generate_synthetic_data(200, 15);
    
    for &l1_penalty in [0.0, 0.01, 0.1].iter() {
        for &l2_penalty in [0.0, 0.01, 0.1].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("l1_{}_l2_{}", l1_penalty, l2_penalty)),
                &(l1_penalty, l2_penalty),
                |b, &(l1_penalty, l2_penalty)| {
                    b.iter(|| {
                        let mut model = CoxModel::new()
                            .with_l1_penalty(l1_penalty)
                            .with_l2_penalty(l2_penalty)
                            .with_max_iterations(200)
                            .with_tolerance(1e-4);
                        model.fit(black_box(&data)).unwrap();
                    });
                },
            );
        }
    }
    group.finish();
}

fn benchmark_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    
    let train_data = generate_synthetic_data(200, 10);
    let mut model = CoxModel::new();
    model.fit(&train_data).unwrap();
    
    for &n_samples in [50, 100, 500, 1000].iter() {
        let test_data = generate_synthetic_data(n_samples, 10);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_samples", n_samples)),
            &n_samples,
            |b, &_n_samples| {
                b.iter(|| {
                    model.predict(black_box(test_data.covariates())).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn benchmark_metrics_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics");
    
    let data = generate_synthetic_data(300, 8);
    let mut model = CoxModel::new();
    model.fit(&data).unwrap();
    let risk_scores = model.predict(data.covariates()).unwrap();
    
    group.bench_function("c_index", |b| {
        b.iter(|| {
            cox_hazards::metrics::concordance_index(
                black_box(risk_scores.view()),
                black_box(data.times()),
                black_box(data.events()),
            ).unwrap();
        });
    });
    
    group.bench_function("harrell_c_index", |b| {
        b.iter(|| {
            cox_hazards::metrics::harrell_c_index(
                black_box(risk_scores.view()),
                black_box(data.times()),
                black_box(data.events()),
            ).unwrap();
        });
    });
    
    group.bench_function("log_likelihood", |b| {
        b.iter(|| {
            cox_hazards::metrics::log_partial_likelihood(
                black_box(&data),
                black_box(risk_scores.view()),
            ).unwrap();
        });
    });
    
    group.bench_function("all_metrics", |b| {
        b.iter(|| {
            cox_hazards::metrics::ModelMetrics::compute(
                black_box(&data),
                black_box(risk_scores.view()),
                8,
            ).unwrap();
        });
    });
    
    group.finish();
}

fn benchmark_data_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_operations");
    
    for &n_samples in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_samples", n_samples)),
            &n_samples,
            |b, &n_samples| {
                b.iter(|| {
                    generate_synthetic_data(black_box(n_samples), 10);
                });
            },
        );
    }
    
    // Test data subset operations
    let large_data = generate_synthetic_data(1000, 10);
    let indices: Vec<usize> = (0..500).collect();
    
    group.bench_function("subset_500_from_1000", |b| {
        b.iter(|| {
            large_data.subset(black_box(&indices)).unwrap();
        });
    });
    
    // Test standardization
    let test_data = generate_synthetic_data(500, 15);
    group.bench_function("standardize_covariates", |b| {
        b.iter(|| {
            let mut data_copy = test_data.clone();
            data_copy.standardize_covariates().unwrap();
            black_box(data_copy);
        });
    });
    
    group.finish();
}

fn benchmark_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale");
    group.sample_size(10); // Reduce sample size for large benchmarks
    
    // Large sample size
    let large_data = generate_synthetic_data(2000, 5);
    group.bench_function("2000_samples_5_features", |b| {
        b.iter(|| {
            let mut model = CoxModel::new()
                .with_max_iterations(50)
                .with_tolerance(1e-3);
            model.fit(black_box(&large_data)).unwrap();
        });
    });
    
    // High dimensional
    let high_dim_data = generate_synthetic_data(200, 50);
    group.bench_function("200_samples_50_features", |b| {
        b.iter(|| {
            let mut model = CoxModel::new()
                .with_l1_penalty(0.1)  // Use regularization for high-dim
                .with_max_iterations(100)
                .with_tolerance(1e-3);
            model.fit(black_box(&high_dim_data)).unwrap();
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_cox_fitting,
    benchmark_elastic_net_fitting,
    benchmark_prediction,
    benchmark_metrics_computation,
    benchmark_data_operations,
    benchmark_large_scale
);

criterion_main!(benches);