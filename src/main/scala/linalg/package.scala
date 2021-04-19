package org.me

import breeze.linalg.{Axis, DenseMatrix, DenseVector, NotConvergedException, diag, kron, lowerTriangular, max, qr, sum, svd}
import breeze.numerics.{abs, log}
import breeze.stats.mean
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW
import spire.implicits.cfor

import scala.annotation.tailrec

package object linalg {

  /**
   * Calculate ax + y for constant a, vector x and vector y
   *
   * @param a : Constant double.
   * @param x : DenseVector x.
   * @param y : DenseVector y.
   * @return ax + y
   */
  def axpy(a: Double,
           x: DenseVector[Double],
           y: DenseVector[Double],
           copy: Boolean): DenseVector[Double] = {
    val yc = if (copy) y.copy else y
    blas.daxpy(x.length, a, x.data, 1, yc.data, 1)
    yc
  }

  /**
   * Compare two matrices element wise and return true if all elements are close.
   *
   * @param X : DenseMatrix
   * @param Y : DenseMatrix
   * @return True if all elements of X and Y are close.
   */
  def allClose(X: DenseMatrix[Double], Y: DenseMatrix[Double], tol: Double = 1E-5): Boolean = {
    (abs(X - Y) <:< tol).toDenseVector.reduce((x, y) => x & y)
  }

  /**
   * Backsolve an upper-triangular linear system
   * with a single RHS
   *
   * @param A An upper-triangular matrix
   * @param y A single vector RHS
   * @return The solution, x, of the linear system A x = y
   */
  def backSolve(A: DenseMatrix[Double],
                y: DenseVector[Double],
                copy: Boolean): DenseVector[Double] = {
    val yc = if (copy) y.copy else y
    blas.dtrsv("U", "N", "N", A.cols, A.toArray,
      A.rows, yc.data, 1)
    yc
  }

  /**
   * Forwardsolve an Lower-triangular linear system
   * with a single RHS
   *
   * @param A An lower-triangular matrix
   * @param y A single vector RHS
   * @return The solution, x, of the linear system A x = y
   */
  def forwardSolve(A: DenseMatrix[Double],
                   y: DenseVector[Double],
                   copy: Boolean): DenseVector[Double] = {
    val yc = if (copy) y.copy else y
    blas.dtrsv("L", "N", "N", A.cols, A.toArray,
      A.rows, yc.data, 1)
    yc
  }

  /**
   * Backsolve an upper-triangular linear system
   * with multiple RHSs
   *
   * @param A An upper-triangular matrix
   * @param Y A matrix with columns corresponding to RHSs
   * @return Matrix of solutions, X, to the linear system A X = Y
   */
  def backSolve(A: DenseMatrix[Double],
                Y: DenseMatrix[Double],
                copy: Boolean): DenseMatrix[Double] = {
    val yc = if (copy) Y.copy else Y
    blas.dtrsm("L", "U", "N", "N", yc.rows, yc.cols, 1.0, A.toArray, A.rows, yc.data, yc.rows)
    yc
  }

  /**
   * Forwardsolve an upper-triangular linear system
   * with multiple RHSs
   *
   * @param A An lower-triangular matrix
   * @param Y A matrix with columns corresponding to RHSs
   * @return Matrix of solutions, X, to the linear system A X = Y
   */
  def forwardSolve(A: DenseMatrix[Double],
                   Y: DenseMatrix[Double],
                   copy: Boolean): DenseMatrix[Double] = {
    val yc = if (copy) Y.copy else Y
    blas.dtrsm("L", "L", "N", "N", yc.rows, yc.cols, 1.0, A.toArray, A.rows, yc.data, yc.rows)
    yc
  }

  def inplaceLowerTriangular(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    val N = X.rows
    cfor(0)(_ < X.rows, _ + 1) {
      i => {
        X(i, (i + 1) until X.rows) := 0.0
      }
    }
    X
  }


  /**
   * Implementation of cholesky decomposition, replacing the breeze implementation
   * to get rid of the checks for X being symmetric.
   *
   * @param X : Matrix to decompose, assumed symmetric and non-empty. Only lower values are used.
   * @return The lower triangular in the cholesky decomposition of X such that X = LLt.
   */
  def cholesky(X: DenseMatrix[Double],
               copy: Boolean): DenseMatrix[Double] = {

    val N = X.rows
    val A = if (copy) lowerTriangular(X) else inplaceLowerTriangular(X)
    val info = new intW(0)
    lapack.dpotrf(
      "L" /* lower triangular */ ,
      N /* number of rows */ ,
      A.data,
      scala.math.max(1, N) /* LDA */ ,
      info
    )
    if (info.`val` > 0)
      throw new NotConvergedException(NotConvergedException.Iterations)
    A
  }

  /**
   * Implementation of cholesky decomposition but adding jitter to diagonal to ensure psd.
   *
   * @param X       Possibly close to Psd matrix X.
   * @param jit     Jitter to add while trying.
   * @param jit_max Maximum jitter to use on matrix X before failing.
   * @return The lower triangular in the cholesky decomposition of X such that X = LLt.
   */
  @tailrec
  def jitChol(X: DenseMatrix[Double],
              jit: Double = 1e-16,
              jit_max: Double = 1e-3): DenseMatrix[Double] = {
    try {
      cholesky(X, true)
    } catch {
      case e: NotConvergedException =>
        if (jit > jit_max) {
          throw e
        } else {
          val diagDelta = jit * mean(diag(X))
          jitChol(X + diagDelta * DenseMatrix.eye[Double](X.rows), jit = 10.0 * jit, jit_max = jit_max, copy = true)
        }
    }
  }

  /**
   * Calculate SVD of matrix X and retain only singular values of sufficient size.
   *
   * @param X   Matrix to decompose.
   * @param tol Tolerance for singular values.
   * @return left singular vectors, singular values, right singular vectors.
   */
  def reducedRankSvd(X: DenseMatrix[Double],
                     tol: Double = 1e-16): (DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double]) = {
    val decomp = svd.reduced(X)
    val nKeep = decomp.singularValues.toArray.toList.indexWhere(_ < (decomp.singularValues(0) * scala.math.sqrt(tol)))
    val indsKeep = if (nKeep == -1) 0 until decomp.singularValues.size else 0 until nKeep
    val U = decomp.leftVectors(::, indsKeep).toDenseMatrix
    val d = decomp.singularValues(indsKeep).toDenseVector
    val Vt = decomp.rightVectors(indsKeep, ::).toDenseMatrix
    (U, d, Vt)
  }

  /**
   * Check if a matrix is diagonal by checking that removing the diagonal from the matrix results in the
   * zero matrix.
   *
   * @param X Matrix to test if diagonal.
   * @return Boolean True if X is diagonal.
   */
  def isDiagonal(X: DenseMatrix[Double]): Boolean = sum(X - diag(diag(X))) == 0

  /**
   * Calculate eigenvalues and right eigenvectors of symmetric matrix using LAPACK dsyev.
   *
   * Eigenvalues and vectors returned in ascending value.
   *
   * @param X DenseMatrix.
   * @return Tuple containing eigenvalues and right eigen vectors.
   */
  def eigh(X: DenseMatrix[Double], copy: Boolean): (DenseVector[Double], DenseMatrix[Double]) = {
    val A = if (copy) X.copy else X
    val N = A.rows
    val evs = DenseVector.zeros[Double](N)
    val lwork = scala.math.max(1, 3 * N - 1)
    val work = Array.ofDim[Double](lwork)
    val info = new intW(0)
    lapack.dsyev(
      "V" /* eigenvalues & eigenvectors "V" */ ,
      "L" /* lower triangular */ ,
      N /* number of rows */ ,
      A.data,
      scala.math.max(1, N) /* LDA */ ,
      evs.data,
      work /* workspace */ ,
      lwork /* workspace size */ ,
      info
    )
    assert(info.`val` >= 0)

    if (info.`val` > 0)
      throw new NotConvergedException(NotConvergedException.Iterations)

    (evs, A)
  }

  /** *
   * Solve a linear system Ax=B where A is symmetric pd using the cholesky decomp of A named L.
   *
   * @param L Cholesky lower triangle of A.
   * @param B RHS of the linear system.
   * @return x matrix which solves Ax=B.
   */
  def choSolve(L: DenseMatrix[Double],
               B: DenseMatrix[Double],
               copy: Boolean): DenseMatrix[Double] = {
    val N = L.rows
    val NRHS = B.cols
    val Bc = if (copy) B.copy else B
    val info = new intW(0)
    lapack.dpotrs(
      "L", /*lower triangular*/
      N, /*number of rows */
      NRHS, /*number of columns of B*/
      L.data, /* Lower triangular factor to solve with */
      scala.math.max(1, N), /*Leading dimension of A*/
      Bc.data, /*RHS of solve, overwrites as solution.*/
      scala.math.max(1, N), /*Leading dimension of B*/
      info
    )
    assert(info.`val` >= 0)

    if (info.`val` > 0)
      throw new NotConvergedException(NotConvergedException.Iterations)
    Bc
  }

  /** *
   * Solve a linear system Ax=b where A is symmetric pd using the cholesky decomp of A named L.
   *
   * @param L Cholesky lower triangle of A.
   * @param b RHS of the linear system.
   * @return x matrix which solves Ax=B.
   */
  def choSolve(L: DenseMatrix[Double],
               b: DenseVector[Double],
               copy: Boolean): DenseVector[Double] = {
    val N = L.rows
    val NRHS = 1
    val bc = if (copy) b.copy else b
    val info = new intW(0)
    lapack.dpotrs(
      "L", /*lower triangular*/
      N, /*number of rows */
      NRHS, /*number of columns of B*/
      L.data, /* Lower triangular factor to solve with */
      scala.math.max(1, N), /*Leading dimension of A*/
      bc.data, /*RHS of solve, overwrites as solution.*/
      scala.math.max(1, N), /*Leading dimension of B*/
      info
    )
    assert(info.`val` >= 0)

    if (info.`val` > 0)
      throw new NotConvergedException(NotConvergedException.Iterations)
    bc
  }

  /**
   * Obtain the null space of a constraint matrix.
   *
   * @param C Constraints.
   * @return Matrix which corresponds to the null space of constrain matrix C.
   */
  def nullSpaceConstraint(C: DenseMatrix[Double]): DenseMatrix[Double] = {
    val n = C.rows
    val QR = qr(C.t)
    val cols = n until QR.q.cols
    QR.q(::, cols).toDenseMatrix
  }

  /**
   * Kronecker product over columns of dense matrix only.
   *
   * @param a DenseMatrix
   * @param b DenseMatrix
   * @return Return column wise kronecker product of a and b.
   */
  def columnKron(a: DenseMatrix[Double], b: DenseMatrix[Double]): DenseMatrix[Double] = {
    val n = a.rows
    require(n == b.rows)
    val res = DenseMatrix.zeros[Double](n, a.cols * b.cols)
    cfor(0)(_ < n, _ + 1) {
      i => {
        val c = kron(a(i, ::).t.toDenseMatrix, b(i, ::).t.toDenseMatrix)
        res(i, ::) := c.toDenseVector.t
      }
    }
    res
  }

  /**
   * Return the infinity norm of a matrix.
   *
   * @param X Dense Matrix
   * @return Infinity norm of matrix X.
   */
  def infNorm(X: DenseMatrix[Double]): Double = {
    max(sum(abs(X), Axis._1))
  }

  /**
   * Return the one norm of a matrix.
   *
   * @param X Dense Matrix
   * @return Infinity norm of matrix X.
   */
  def oneNorm(X: DenseMatrix[Double]): Double = {
    max(sum(abs(X), Axis._0))
  }

  /**
   * Calculate the log determinant of A from cholesky decomposition where A = LLt
   *
   * @param L Lower decomposition of A.
   * @return
   */
  def detFromChol(L: DenseMatrix[Double]): Double = {
    2.0 * sum(log(diag(L)))
  }

  /**
   * Inplace turn lower/upper matrix into symmetric matrix.
   *
   * @param L Upper or lower triangular matrix.
   * @return
   */
  def symFromUPLO(L: DenseMatrix[Double],
                  copy: Boolean = true): DenseMatrix[Double] = {
    val Lc = if (copy) L.copy else L
    Lc :+= Lc.t
    diag(Lc) :*= 0.5
    Lc
  }

  /**
   * Calculate empirical covariance of matrix.
   *
   * @param A  data matrix each column is a value of the random vector.
   * @param df degrees of freedom to use in divisor of cov.
   * @return
   */
  def cov(A: DenseMatrix[Double], df: Int): DenseMatrix[Double] = {
    require(df >= 0)
    val n = A.cols
    val ndf = 1.0 / (n - df)
    val D: DenseMatrix[Double] = A.copy
    val mu: DenseVector[Double] = sum(D, Axis._1) *:* ndf
    cfor(0)(_ < n, _ + 1) {
      i => D(::, i) :-= mu
    }
    val C = (D * D.t) *:* ndf
    (C + C.t) *:* 0.5
  }

  /**
   * Calculate empirical covariance of matrix with known mean.
   *
   * @param A  data matrix each column is a value of the random vector.
   * @param mu mean vector.
   * @param df degrees of freedom to use in divisor of cov.
   * @return
   */
  def cov(A: DenseMatrix[Double], mu: DenseVector[Double], df: Int): DenseMatrix[Double] = {
    require(df >= 0)
    require(mu.length == A.rows)
    val n = A.cols
    val ndf = 1.0 / (n - df)
    val D: DenseMatrix[Double] = A.copy
    cfor(0)(_ < n, _ + 1) {
      i => D(::, i) :-= mu
    }
    val C = (D * D.t) *:* ndf
    (C + C.t) *:* 0.5
  }
}
