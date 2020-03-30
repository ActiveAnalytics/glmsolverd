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
  
  writeln("Timed test for Poisson RNG, n = ", n, ", p = ", 
          p, ", time(ms): ", sw.peek.total!"msecs");

  sw.stop();
}
