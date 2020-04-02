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
import glmsolverd.sample;

import std.stdio: writeln;
import std.parallelism;
import std.range: iota;
import std.array: array;
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
}


