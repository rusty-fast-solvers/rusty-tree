use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusty_tree::adaptive_octree::*;
use rand::Rng;



fn benchmark_make_adaptive_octree(c: &mut Criterion) {
    let nparticles = 1000000;
    let max_particles = 200;
    let mut rng = rand::thread_rng();
    let mut particles = ndarray::Array2::<f64>::zeros((3, nparticles));
    particles.map_inplace(|item| *item = rng.gen::<f64>());


    c.bench_function("create adaptive octree", |b| {
        b.iter(|| {
            adaptive_octree(particles.view(), black_box(max_particles));
        })
    });


}



criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30).measurement_time(std::time::Duration::from_secs(10));
    targets = benchmark_make_adaptive_octree,
            }
criterion_main!(benches);
