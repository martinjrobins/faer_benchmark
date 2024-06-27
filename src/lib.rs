use nalgebra_sparse::{io::load_coo_from_matrix_market_file, CscMatrix};




pub fn load_matrix(filename: &str) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<usize>, Vec<f64>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    
    use faer::sparse::linalg::solvers::SymbolicLu;
    use faer::sparse::{linalg::solvers::Lu, SparseColMat, SymbolicSparseColMat};
    use faer::Col;
    use faer::prelude::SpSolver;
    use suitesparse_sys::{
        klu_l_analyze as klu_analyze, klu_l_common as klu_common, klu_l_defaults as klu_defaults, klu_l_factor as klu_factor, klu_l_free_numeric as klu_free_numeric, klu_l_free_symbolic as klu_free_symbolic,
        klu_l_solve as klu_solve,
    };

    #[test]
    fn it_works() {

        let (nrows, ncols, mut col_offsets, nnz_per_col, mut row_indices, mut values) = load_matrix("heat2d_5.mtx");
        let symbolic = SymbolicSparseColMat::new_checked(
            nrows,
            ncols,
            col_offsets.clone(),
            Some(nnz_per_col),
            row_indices.clone(),
        );
        let matrix = SparseColMat::<usize, f64>::new(symbolic, values.clone());
        let mut x = Col::from_fn(nrows, |_i| 1.0);
        let symbolic =
            SymbolicLu::try_new(matrix.symbolic()).expect("Failed to create symbolic LU");
        let lu = Lu::try_new_with_symbolic(symbolic, matrix.as_ref())
            .expect("Failed to factorise matrix");
        lu.solve_in_place(x.as_mut());
        let x_faer: Vec<f64> = x.as_slice().to_vec();
        
        let mut common = klu_common::default();
        unsafe { klu_defaults(&mut common) };
        let n = nrows as i64;
        let mut symbolic = unsafe { klu_analyze(n, col_offsets.as_mut_ptr() as *mut i64, row_indices.as_mut_ptr() as *mut i64, &mut common) };
        let mut numeric = unsafe {
            klu_factor(
                col_offsets.as_mut_ptr() as *mut i64,
                row_indices.as_mut_ptr() as *mut i64,
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
        
        for i in 0..nrows {
            assert!((x_faer[i] - x[i]).abs() < 1e-10);
        }
        

    }
}
