package org.me
package linalg

import breeze.linalg.{DenseMatrix, DenseVector, diag, isClose, lowerTriangular}
import breeze.stats.distributions.Gaussian
import org.scalatest.FunSuite

import scala.util.Random.nextDouble

class packageTest extends FunSuite {
  test("testAxpy") {
    val a = nextDouble()
    val x = DenseVector.rand[Double](1, Gaussian(0, 1))
    val y = DenseVector.rand[Double](1, Gaussian(0, 1))
    assert(isClose(axpy(a, x, y, copy = true), a * x + y))
  }

  test("testIsDiagonal") {
    val d = DenseVector.rand[Double](5)
    assert(isDiagonal(diag(d)))
    val nd = DenseMatrix.rand[Double](5, 5)
    assert(!isDiagonal(nd))

  }

  test("testJitChol") {
    val X = DenseMatrix((1.0, 0.0), (0.0, 0.0))
    val L = jitChol(X, jit_max = 1e-8, copy = true)
    assert(allClose(L, cholesky(DenseMatrix((1.0, 0.0), (0.0, 1e-18)), copy = true)))
  }

  test("testNullSpaceConstraint") {
    val C = DenseMatrix.ones[Double](1, 2)
    assert(allClose(nullSpaceConstraint(C),
      DenseMatrix(-0.5 * scala.math.sqrt(2), 0.5 * scala.math.sqrt(2))))
  }

  test("testReducedRankSvd") {
    val D = diag(DenseVector(1.0, 2.0, 3.0, 4.0, 0.0))
    val rrSvD = reducedRankSvd(D)
    assert(rrSvD._2.size == 4)
  }

  test("testInplaceToLower") {
    val X = DenseMatrix.rand[Double](5, 5)
    val L = inplaceLowerTriangular(X.copy)
    val LL = lowerTriangular(X)
    assert(allClose(L, LL))
  }

  test("testCholesky") {
    val sqrtX = DenseMatrix.rand[Double](5, 5)
    val X = sqrtX.t * sqrtX
    val L = cholesky(X, copy = false)
    assert(allClose(L * L.t, sqrtX.t * sqrtX))
  }

  test("testEigh") {
    val d = DenseVector(1.0, 2.0, 3.0, 4.0)
    val D = diag(d)
    val (eigenValues, eigenVectors) = eigh(D, copy = true)
    assert(isClose(eigenValues, d))
    assert(allClose(eigenVectors, DenseMatrix.eye[Double](4)))
  }

  test("testChoSolve") {
    val sqrtA = DenseMatrix.rand[Double](5, 5)
    val A = sqrtA.t * sqrtA
    val x = DenseVector.rand[Double](5)
    val b = A * x
    assert(isClose(choSolve(cholesky(A, copy = true), b.toDenseMatrix.t, copy = true).toDenseVector, x))
  }

  test("testChoSolve Single RHS") {
    val sqrtA = DenseMatrix.rand[Double](5, 5)
    val A = sqrtA.t * sqrtA
    val x = DenseVector.rand[Double](5)
    val b = A * x
    assert(isClose(choSolve(cholesky(A, copy = true), b, copy = true), x))
  }

  test("testColumnKron") {
    val X1 = DenseMatrix((1.0, 2.0), (2.0, 3.0))
    val X2 = DenseMatrix((3.0, 3.0), (4.0, 5.0))
    val X = columnKron(X1, X2)
    val XX = DenseMatrix(
      (1.0 * 3.0, 1.0 * 3.0, 2.0 * 3.0, 2.0 * 3.0),
      (2.0 * 4.0, 2.0 * 5.0, 3.0 * 4.0, 3.0 * 5.0))
    assert(allClose(X, XX))
  }

  test("testBackSolve") {
    val sqrtA = DenseMatrix.rand[Double](5, 5)
    val A = sqrtA.t * sqrtA
    val U = jitChol(A, copy = true).t
    val x = DenseVector.rand[Double](5)
    val b = U * x
    assert(isClose(backSolve(U, b, copy = true), x))

  }

  test("testBackSolve multiple RHS") {
    val sqrtA = DenseMatrix.rand[Double](5, 5)
    val A = sqrtA.t * sqrtA
    val U = jitChol(A, copy = true).t
    val X = DenseMatrix.rand[Double](5, 5)
    val b = U * X
    assert(allClose(backSolve(U, b, copy = true), X))
  }

}
