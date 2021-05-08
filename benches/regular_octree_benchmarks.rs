use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusty_tree::morton::*;
use rusty_tree::octree::*;
use rand::Rng;


fn benchmark_morton_encoding(c: &mut Criterion) {
    let nparticles = 1000000;
    let level = 5;
    let mut rng = rand::thread_rng();
    let mut particles = ndarray::Array2::<f64>::zeros((3, nparticles));
    particles.map_inplace(|item| *item = rng.gen::<f64>());

    let origin = [0.0; 3];
    let diameter = [1.0; 3];

    c.bench_function("morton encoding", |b| {
        b.iter(|| {
            encode_points(particles.view(), black_box(level), &origin, &diameter);
        })
    });


}

fn benchmark_make_regular_octree(c: &mut Criterion) {
    let nparticles = 1000000;
    let level = 5;
    let mut rng = rand::thread_rng();
    let mut particles = ndarray::Array2::<f64>::zeros((3, nparticles));
    particles.map_inplace(|item| *item = rng.gen::<f64>());


    c.bench_function("create regular octree", |b| {
        b.iter(|| {
            regular_octree(particles.view(), black_box(level));
        })
    });


}



criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30).measurement_time(std::time::Duration::from_secs(10));
    targets = benchmark_morton_encoding,
              benchmark_make_regular_octree,
            }
criterion_main!(benches);
