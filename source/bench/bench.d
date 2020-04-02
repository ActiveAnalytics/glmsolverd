module bench.bench;

import std.stdio: writeln;
import std.typecons: Tuple, tuple;
import std.algorithm.iteration: mean;
import std.datetime.stopwatch: AutoStart, StopWatch;

/*
  Simple function for doing benchmarking
  ulong n is the number of times the bench should be run for.
  string units is the time units for the StopWatch.
  ulong minN is minimum number of times the benchmark is run
        before the standard deviation is calculated.
  bool doPrint whether the results should be printed or not
*/
auto bench(alias fun, string units = "msecs",
          ulong minN = 10, bool doPrint = false)(ulong n, string msg = "")
{
  auto times = new double[n];
  auto sw = StopWatch(AutoStart.no);
  for(ulong i = 0; i < n; ++i)
  {
    sw.start();
    fun();
    sw.stop();
    times[i] = cast(double)sw.peek.total!units;
    sw.reset();
  }
  double ave = mean(times);
  double sd = 0;

  if(n >= minN)
  {
    for(ulong i = 0; i < n; ++i)
      sd += (times[i] - ave)^^2;
    sd /= (n - 1);
    sd ^^= 0.5;
  }else{
    sd = double.nan;
  }

  static if(doPrint)
    writeln(msg ~ "Mean time("~ units ~ "): ", ave, ", Standard Deviation: ", sd);

  return tuple!("mean", "sd")(ave, sd);
}

