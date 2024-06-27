use criterion::{criterion_group, criterion_main, Criterion};
use faer::sparse::linalg::solvers::SymbolicLu;
use faer::sparse::{linalg::solvers::Lu, SparseColMat, SymbolicSparseColMat};
use faer::Col;
use faer::prelude::SpSolver;
use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CscMatrix};
use suitesparse_sys::{
    klu_analyze, klu_common, klu_defaults, klu_factor, klu_free_numeric, klu_free_symbolic,
    klu_solve,
};

fn load_matrix(filename: &str) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<usize>, Vec<f64>) {
    let coo = load_coo_from_matrix_market_file(filename).unwrap();
    let csc = CscMatrix::from(&coo);
    let nrows = csc.nrows();
    let ncols = csc.ncols();
    let nnz_per_col = csc.col_iter().map(|col| col.nnz()).collect::<Vec<_>>();
    let (col_offsets, row_indices, values) = csc.csc_data();
    let col_offsets = col_offsets.to_vec();
    let row_indices = row_indices.to_vec();
    let values = values.to_vec();
    (nrows, ncols, col_offsets, nnz_per_col, row_indices, values)
}

fn criterion_benchmark(c: &mut Criterion) {
    let filenames = ["heat2d_5.mtx", "heat2d_10.mtx", "heat2d_20.mtx", "heat2d_30.mtx"];
    for filename in filenames {
        c.bench_function(format!("{}_symbolic_faer", filename).as_str(), |b| {
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
                SymbolicLu::try_new(matrix.symbolic()).expect("Failed to create symbolic LU");
            })
        });
        c.bench_function(format!("{}_symbolic_klu", filename).as_str(), |b| {
            let (nrows, _ncols, mut col_ptrs, _nnz_per_col, mut row_ind, _values) =
                load_matrix(filename);

            let mut common = klu_common::default();
            unsafe { klu_defaults(&mut common) };
            let n = nrows as i32;
            
            b.iter(|| {
                let mut symbolic = unsafe { klu_analyze(n, col_ptrs.as_mut_ptr() as *mut i32, row_ind.as_mut_ptr() as *mut i32, &mut common) };
                unsafe { klu_free_symbolic(&mut symbolic, &mut common) };
            })
        });
    }
    for filename in filenames {
        c.bench_function(format!("{}_numeric_faer", filename).as_str(), |b| {
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
            let symbolic =
                SymbolicLu::try_new(matrix.symbolic()).expect("Failed to create symbolic LU");
            b.iter(|| {
                Lu::try_new_with_symbolic(symbolic.clone(), matrix.as_ref())
                    .expect("Failed to factorise matrix");
            })
        });
        c.bench_function(format!("{}_numeric_klu", filename).as_str(), |b| {
            let (nrows, _ncols, mut col_ptrs, _nnz_per_col, mut row_ind, mut values) =
                load_matrix(filename);

            let mut common = klu_common::default();
            unsafe { klu_defaults(&mut common) };
            let n = nrows as i32;
            let mut symbolic = unsafe { klu_analyze(n, col_ptrs.as_mut_ptr() as *mut i32, row_ind.as_mut_ptr() as *mut i32, &mut common) };
            b.iter(|| {
                let mut numeric = unsafe {
                    klu_factor(
                        col_ptrs.as_mut_ptr() as *mut i32,
                        row_ind.as_mut_ptr() as *mut i32,
                        values.as_mut_ptr(),
                        symbolic,
                        &mut common,
                    )
                };
                unsafe { klu_free_numeric(&mut numeric, &mut common) };
            });
            unsafe { klu_free_symbolic(&mut symbolic, &mut common) };
        });
    }
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
            let symbolic =
                SymbolicLu::try_new(matrix.symbolic()).expect("Failed to create symbolic LU");
            let lu = Lu::try_new_with_symbolic(symbolic.clone(), matrix.as_ref())
                    .expect("Failed to factorise matrix");
            let x = Col::from_fn(nrows, |i| i as f64);
            b.iter(|| {
                let x = x.clone();
                lu.solve_in_place(x);
            })
        });
        c.bench_function(format!("{}_solve_klu", filename).as_str(), |b| {
            let (nrows, _ncols, mut col_ptrs, _nnz_per_col, mut row_ind, mut values) =
                load_matrix(filename);

            let mut common = klu_common::default();
            unsafe { klu_defaults(&mut common) };
            let n = nrows as i32;
            let mut symbolic = unsafe { klu_analyze(n, col_ptrs.as_mut_ptr() as *mut i32, row_ind.as_mut_ptr() as *mut i32, &mut common) };
            let mut numeric = unsafe {
                klu_factor(
                    col_ptrs.as_mut_ptr() as *mut i32,
                    row_ind.as_mut_ptr() as *mut i32,
                    values.as_mut_ptr(),
                    symbolic,
                    &mut common,
                )
            };
            let n = nrows as i32;
            let mut x = vec![1.0; nrows];
            b.iter(|| {
                unsafe { klu_solve(symbolic, numeric, n, 1, x.as_mut_ptr(), &mut common) };
                
            });
            unsafe { klu_free_symbolic(&mut symbolic, &mut common) };
            unsafe { klu_free_numeric(&mut numeric, &mut common) };
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
