module demos.scratch;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;
import glmsolverd.linearalgebra;
import glmsolverd.io;
import glmsolverd.fit;
import glmsolverd.lbfgs;
import glmsolverd.sample;

import std.stdio: writeln;
import std.parallelism;
import std.range: iota;
import std.datetime.stopwatch : AutoStart, StopWatch;

/*
  Informal testing and diagnosis
*/

/* Testing out poisson RNG */
void poissonRNG(ulong n, ulong p)
{
  auto sw = StopWatch(AutoStart.no);
  sw.start();

  ulong seed = 3;
  AbstractDistribution!(double) distrib  = new PoissonDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto poissonData = simulateData!(double)(distrib, link, n, p, seed);
  auto poissonX = poissonData.X;
  auto poissonY = poissonData.y;
  sw.stop();
  
  writeln("Timed test for Poisson RNG, n = ", n, ", p = ", 
          p, ", time(ms): ", sw.peek.total!"msecs");
}

/* Test append prepend Column */
void appendDemo()
{
  import std.array: array;
  double[] x = [1.0, 2, 3, 4, 
      5, 6, 7, 8, 9, 10, 11, 12];
  auto mat = new Matrix!(double)(x, [4, 3]);
  auto vec = new ColumnVector!(double)([13.0, 14, 15, 16]);
  writeln("Original matrix: ", mat);
  mat.appendColumn(vec);
  writeln("Appended matrix: ", mat);

  mat = new Matrix!(double)(x, [4, 3]);
  mat.prependColumn(vec);
  writeln("Prepend matrix: ", mat);

  mat = mat.columnSelect(1, 3);
  writeln("Selected Columns: ", mat);

  double[] z = iota!(double)(0.0, 35.0, 1.0).array;
  writeln("Array length: ", z.length);
  mat = new Matrix!(double)(z, [7, 5]);
  writeln("Matrix: ", mat);
  mat = mat.columnSelect(0, 3);
  writeln("Selected Columns [0..3]: ", mat);
  writeln("Select the 2nd column: \n", mat.columnSelect(1).array, "\n\n");

  /* Test removing column from the matrix */
  writeln("************* Begin removing columns from a matrix test *************\n");
  z = iota!(double)(0.0, 35.0, 1.0).array;
  mat = new Matrix!(double)(z, [7, 5]);
  writeln("Current matrix: ", mat);
  writeln("Remove first column: ", mat.refColumnRemove(0));
  z = iota!(double)(0.0, 35.0, 1.0).array;
  mat = new Matrix!(double)(z, [7, 5]);
  writeln("After refreshing remove end column: ", mat.refColumnRemove(4));
  z = iota!(double)(0.0, 35.0, 1.0).array;
  mat = new Matrix!(double)(z, [7, 5]);
  writeln("After refreshing remove 3rd column: ", mat.refColumnRemove(2));
  z = iota!(double)(0.0, 35.0, 1.0).array;
  mat = new Matrix!(double)(z, [7, 5]);
  mat.refColumnAssign(42.0, 1);
  writeln("Replace the 2nd column with 42: ", mat);
  mat.refColumnAssign(fillColumn!(double)(18.0, 7), 2);
  writeln("Replace the 3rd column with 18: ", mat);
  writeln("************* End removing columns from a matrix test *************\n\n");
}


void lbfgsComponentTest()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  //auto gammaData = simulateData!(double)(distrib, link, 10_000, 30, seed);
  auto gammaData = simulateData!(double)(distrib, link, 100, 10, seed);
  auto gammaX = gammaData.X;
  auto gammaY = new ColumnVector!(double)(gammaData.y.array);
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = vectorToBlock(gammaY, 10);
  
  import std.algorithm.iteration: mean;
  ulong p = gammaX.ncol; ulong n = gammaX.nrow;
  auto coef = createRandomColumnVector!(double)(p);
  /* In gradient decent we can use this as the intercept */
  double intercept = link.linkfun(mean(gammaY.array));
  coef[0] = intercept;

  /* Basic testing for the Deviance class works */
  auto deviance = new Deviance!(double);
  auto dev = deviance.deviance(new RegularData(), distrib, link, 
              coef, gammaY, gammaX, zerosColumn!(double)(0), 
              zerosColumn!(double)(0));
  writeln("Deviance for regular data: ", dev);

  dev = deviance.deviance(new Block1D(), distrib, link, 
              coef, gammaBlockY, gammaBlockX, new ColumnVector!(double)[0], 
              new ColumnVector!(double)[0]);
  writeln("Deviance for block data: ", dev);
  
  dev = deviance.deviance(new Block1DParallel(), distrib, link, 
              coef, gammaBlockY, gammaBlockX, new ColumnVector!(double)[0], 
              new ColumnVector!(double)[0]);
  writeln("Deviance for parallel block data: ", dev);
  
  /* Basic testing for the DPhi class works */
  auto dir = createRandomColumnVector!(double)(p);
  auto dphi = new DPhi!(double)();
  auto pgrad = dphi.dPhi(new RegularData(), distrib, link,
              0.5, dir, coef, gammaY, gammaX,
              zerosColumn!(double)(0));
  writeln("d_phi_d_alpha using regular data: ", pgrad);
  pgrad = dphi.dPhi(new Block1D(), distrib, link,
              0.5, dir, coef, gammaBlockY, gammaBlockX, 
              new ColumnVector!(double)[0]);
  writeln("d_phi_d_alpha using block data: ", pgrad);
  pgrad = dphi.dPhi(new Block1DParallel(), distrib, link,
              0.5, dir, coef, gammaBlockY, gammaBlockX, 
              new ColumnVector!(double)[0]);
  writeln("d_phi_d_alpha using paralel block data: ", pgrad, "\n");
  
  /* Basic testing for the Gradient class works */
  auto gradient = new Gradient!(double)();
  auto grad = gradient.gradient(new RegularData(), distrib, link,
                coef, gammaY, gammaX, zerosColumn!(double)(0));
  writeln("Gradient using regular data: \n", grad.array);
  grad = gradient.gradient(new Block1D(), distrib, link,
                coef, gammaBlockY, gammaBlockX, 
                new ColumnVector!(double)[0]);
  writeln("Gradient using block data: \n", grad.array);
  grad = gradient.gradient(new Block1DParallel(), 
                distrib, link, coef, gammaBlockY, gammaBlockX, 
                new ColumnVector!(double)[0]);
  writeln("Gradient using block data: \n", grad.array, "\n");
  
  /* Basic testing for Interpolation class works */
  auto interpolation = new Interpolation!(double)();
  auto interp = interpolation.quadratic(5, 4, -4, 9);
  writeln("Testing quadratic interpolation: ", interp);
  interp = interpolation.cubic(0, 5, 4, 9, -4, 6);
  writeln("Testing cubic interpolation: ", interp, "\n");
  
  /* Basic testing for Backtracking Line Search */
  AbstractLineSearch!(double)  ls = new BackTrackingLineSearch!(double)();
  auto nullVector = zerosColumn!(double)(0);
  auto alphaLS = ls.linesearch(new RegularData(), distrib, 
                    link, dir, coef, gammaY, gammaX, nullVector, 
                    nullVector);
  writeln("Linear search output from regular data, alpha: ", alphaLS);
  
  auto nullBlock = new ColumnVector!(double)[0];
  alphaLS = ls.linesearch(new Block1D(), distrib, link, dir, 
                    coef, gammaBlockY, gammaBlockX, nullBlock, 
                    nullBlock);
  writeln("Linear search output from block data, alpha: ", alphaLS);
  
  alphaLS = ls.linesearch(new Block1DParallel(), distrib, link, dir, 
                    coef, gammaBlockY, gammaBlockX, nullBlock, 
                    nullBlock);
  writeln("Linear search output from block data, alpha: ", alphaLS);
}


void lbfgsTest()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  //auto gammaData = simulateData!(double)(distrib, link, 10_000, 30, seed);
  auto gammaData = simulateData!(double)(distrib, link, 1000, 10, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  auto nullVector = zerosColumn!(double)(0);
  auto nullBlock = new ColumnVector!(double)[0];
  auto p = gammaX.ncol;

  auto lbfgs = new LBFGSSolver!(double)(p, 5);
  auto ls = new BackTrackingLineSearch!(double)();
  auto inverse = new GETRIInverse!(double, CblasColMajor)();

  auto glmModel = glm(new RegularData(), gammaX, gammaY, distrib,
        link, new GESVSolver!(double)(), inverse);
  writeln("\nStandard GLMModel using regular data: ", glmModel);

  glmModel = glm(new RegularData(), gammaX, 
       gammaY, distrib, link, lbfgs, ls, inverse);
  writeln("\nLBFGS GLMModel using regular data: ", glmModel);

  lbfgs = new LBFGSSolver!(double)(p, 5);
  ls = new BackTrackingLineSearch!(double)();
  glmModel = glm(new Block1D(), gammaBlockX, 
       gammaBlockY, distrib, link, lbfgs, ls, inverse);
  writeln("\nLBFGS GLMModel with serial block1D data: ", glmModel);

  lbfgs = new LBFGSSolver!(double)(p, 5);
  ls = new BackTrackingLineSearch!(double)();
  glmModel = glm(new Block1DParallel(), gammaBlockX, 
       gammaBlockY, distrib, link, lbfgs, ls, inverse);
  writeln("\nLBFGS GLMModel with parallel block1D data: ", glmModel);
}

