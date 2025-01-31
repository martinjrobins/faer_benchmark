use criterion::{criterion_group, criterion_main, Criterion};
use faer::sparse::linalg::solvers::SymbolicLu;
use faer::sparse::{linalg::solvers::Lu, SparseColMat, SymbolicSparseColMat};
use faer::Col;
use suitesparse_sys::{
    klu_l_analyze as klu_analyze, klu_l_common as klu_common, klu_l_defaults as klu_defaults, klu_l_factor as klu_factor, klu_l_free_numeric as klu_free_numeric, klu_l_free_symbolic as klu_free_symbolic,
    klu_l_solve as klu_solve,
};
use faer_benchmark::load_matrix;
use faer::linalg::solvers::Solve;
use faer::reborrow::Reborrow;



fn criterion_benchmark(c: &mut Criterion) {
    let filenames = ["heat2d_5.mtx", "heat2d_10.mtx", "heat2d_20.mtx", "heat2d_30.mtx", "robertson_ode_10.mtx", "robertson_ode_100.mtx", "robertson_ode_1000.mtx"];
    for filename in filenames {
        c.bench_function(format!("{}_solve_faer", filename).as_str(), |b| {
            let (nrows, ncols, col_offsets, nnz_per_col, row_indices, values) =
                load_matrix(filename);
            let symbolic = SymbolicSparseColMat::new_checked(
                nrows,
                ncols,
                col_offsets,
                Some(nnz_per_col),
                row_indices,
            );
            let matrix = SparseColMat::<usize, f64>::new(symbolic, values);
            
            b.iter(|| {
                let x = Col::from_fn(nrows, |i| i as f64);
                let symbolic =
                    SymbolicLu::try_new(matrix.symbolic()).expect("Failed to create symbolic LU");
                let lu = Lu::try_new_with_symbolic(symbolic, matrix.rb())
                    .expect("Failed to factorise matrix");
                lu.solve_in_place(x);
            })
        });
        c.bench_function(format!("{}_solve_klu", filename).as_str(), |b| {
            let (nrows, _ncols, mut col_ptrs, _nnz_per_col, mut row_ind, mut values) =
                load_matrix(filename);

            b.iter(|| {
                let mut common = klu_common::default();
                unsafe { klu_defaults(&mut common) };
                let n = nrows as i64;
                let mut symbolic = unsafe { klu_analyze(n, col_ptrs.as_mut_ptr() as *mut i64, row_ind.as_mut_ptr() as *mut i64, &mut common) };
                let mut numeric = unsafe {
                    klu_factor(
                        col_ptrs.as_mut_ptr() as *mut i64,
                        row_ind.as_mut_ptr() as *mut i64,
                        values.as_mut_ptr(),
                        symbolic,
                        &mut common,
                    )
                };
                let n = nrows as i64;
                let mut x = vec![1.0; nrows];
                unsafe { klu_solve(symbolic, numeric, n, 1, x.as_mut_ptr(), &mut common) };
                unsafe { klu_free_symbolic(&mut symbolic, &mut common) };
                unsafe { klu_free_numeric(&mut numeric, &mut common) };
                
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
