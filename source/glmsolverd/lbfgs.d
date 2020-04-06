module glmsolverd.lbfgs;

import std.conv: to;
import std.stdio: writeln;
import std.typecons: Tuple, tuple;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import std.parallelism;
import std.range: iota;

import std.math: pow, sgn;
import std.algorithm: min, max;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;
import glmsolverd.linearalgebra;

/* Regression Weights Class*/
class Weights(T, CBLAS_LAYOUT layout = CblasColMajor)
if(isFloatingPoint!T)
{
  public:
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^(-1);
  }
  ColumnVector!(T) W(RegularData dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(Block1D dataType, AbstractDistribution!T distrib,
              AbstractLink!T link, BlockColumnVector!(T) mu, 
              BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) W(Block1DParallel dataType, 
              AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
}

/* XWX Class */
class XWX(T, CBLAS_LAYOUT layout = CblasColMajor)
if(isFloatingPoint!T)
{
  public:
  void XWX(ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
  }
  void XWX(ref Matrix!(T, layout) xwx, 
              ref BlockMatrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w)
  {
    ulong p = x[0].ncol;
    xwx = zerosMatrix!(T, layout)(p, p);
    ulong nBlocks = x.length;
    for(ulong i = 0; i < nBlocks; ++i)
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      xwx += mult_!(T, layout, CblasTrans, CblasNoTrans)(tmp , x[i]);
    }
  }
  void XWX(Block1DParallel dataType, ref Matrix!(T, layout) xwx, 
              ref BlockMatrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w)
  {
    ulong p = x[0].ncol;
    
    xwx = zerosMatrix!(T, layout)(p, p);
    auto XWX = taskPool.workerLocalStorage(zerosMatrix!(T, layout)(p, p));
    
    ulong nBlocks = x.length;
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      XWX.get += mult_!(T, layout, CblasTrans, CblasNoTrans)(tmp , x[i]);
    }
    foreach (_xwx; XWX.toRange)
      xwx += _xwx;
  }
}

/*
  Class for calculating Deviance

  *May have to include mu, eta as reference input 
  parameters
*/
class Deviance(T, CBLAS_LAYOUT layout = CblasColMajor)
if(isFloatingPoint!T)
{
  public:
  T deviance(RegularData dataType, AbstractDistribution!T distrib, 
              AbstractLink!T link, ColumnVector!(T) coef,
              ColumnVector!T y, Matrix!(T, layout) x, 
              ColumnVector!T weights, ColumnVector!T offset)
  {
    auto residuals = zerosColumn!(T)(y.len);
    auto eta = mult_(x, coef);
    
    if(offset.length != 0)
      eta += offset;
    
    auto mu = link.linkinv(eta);

    if(weights.length == 0)
      residuals = distrib.devianceResiduals(mu, y);
    else
      residuals = distrib.devianceResiduals(mu, y, weights);
    
    return sum!(T)(residuals);
  }
  T deviance(Block1D dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, ColumnVector!(T) coef,
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x,
            BlockColumnVector!(T) weights, BlockColumnVector!T offset)
  {
    ColumnVector!(T)[] residuals;
    auto nBlocks = y.length;
    auto eta = new ColumnVector!(T)[nBlocks];

    for(ulong i = 0; i < nBlocks; ++i)
      eta[i] = mult_(x[i], coef);
    
    if(offset.length != 0)
    {
      for(ulong i = 0; i < nBlocks; ++i)
        eta[i] += offset[i];
    }

    auto mu = link.linkinv(eta);

    if(weights.length == 0)
      residuals = distrib.devianceResiduals(mu, y);
    else
      residuals = distrib.devianceResiduals(mu, y, weights);
    
    T dev = 0;
    for(ulong i = 0; i < nBlocks; ++i)
      dev += sum!(T)(residuals[i]);
    
    return dev;
  }
  T deviance(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, ColumnVector!(T) coef,
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x,
            BlockColumnVector!(T) weights, BlockColumnVector!T offset)
  {
    ColumnVector!(T)[] residuals;
    auto nBlocks = y.length;
    auto eta = new ColumnVector!(T)[nBlocks];

    foreach(i; taskPool.parallel(iota(nBlocks)))
      eta[i] = mult_(x[i], coef);
    
    if(offset.length != 0)
    {
      foreach(i; taskPool.parallel(iota(nBlocks)))
        eta[i] += offset[i];
    }
    auto mu = link.linkinv(dataType, eta);
    
    if(weights.length == 0)
      residuals = distrib.devianceResiduals(dataType, mu, y);
    else
      residuals = distrib.devianceResiduals(dataType, mu, y, weights);
    
    T dev = 0;
    auto devStore = taskPool.workerLocalStorage(cast(T)0);
    
    foreach(i; taskPool.parallel(iota(nBlocks)))
      devStore.get += sum!T(residuals[i]);
    foreach (_dev; devStore.toRange)
      dev += _dev;
    
    return dev;
  }
}

/* dPhi_dAlpha */
class DPhi(T, CBLAS_LAYOUT layout = CblasColMajor)
if(isFloatingPoint!T)
{
  private:
  auto gradient_(AbstractDistribution!T distrib, 
      AbstractLink!T link, ColumnVector!T y, Matrix!(T, layout) x, 
      ColumnVector!T mu, ColumnVector!T eta)
  {
    ulong p = x.ncol;
    ulong ni = x.nrow;
    auto grad = zerosColumn!T(p);
    auto tmp = zerosColumn!(T)(ni);
    T numer = 0;
    T X2 = 0;
    for(ulong i = 0; i < ni; ++i)
    {
      numer = y[i] - mu[i];
      X2 += (numer^^2)/distrib.variance(mu[i]);
      tmp[i] = numer/(link.deta_dmu(mu[i], eta[i]) * distrib.variance(mu[i]));
      for(ulong j = 0; j < p; ++j)
      {
        grad[j] += tmp[i] * x[i, j];
      }
    }
    return tuple!("X2", "grad")(X2, grad);
  }

  public:
  T dPhi(RegularData dataType, AbstractDistribution!T distrib, 
              AbstractLink!T link, T alpha, ColumnVector!T dir,
              ColumnVector!(T) _coef, ColumnVector!T y, 
              Matrix!(T, layout) x, ColumnVector!T offset)
  {
    auto coef = _coef.dup;
    coef += alpha*dir;
    auto eta = mult_(x, coef);
    
    if(offset.length != 0)
      eta += offset;
    
    auto mu = link.linkinv(eta);

    ulong p = x.ncol; ulong n = x.nrow;
    auto tmpGrad = gradient_(distrib, link, y, x, mu, eta);
    auto df = cast(T)(n - p);

    assert(df > 0, "Number of items n is not greater than the number of parameters p.");
    T phi = tmpGrad.X2/df;
    auto grad = tmpGrad.grad/phi;
    return dotSum!(T)(grad, dir);
  }
  T dPhi(Block1D dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, T alpha, ColumnVector!T dir,
            ColumnVector!(T) _coef, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!T offset)
  {
    auto coef = _coef.dup;
    coef += alpha*dir;
    ulong nBlocks = y.length; ulong n = 0;
    auto eta = new ColumnVector!(T)[nBlocks];

    for(ulong i = 0; i < nBlocks; ++i)
      eta[i] = mult_(x[i], coef);
    
    if(offset.length != 0)
    {
      for(ulong i = 0; i < nBlocks; ++i)
        eta[i] += offset[i];
    }
    auto mu = link.linkinv(eta);

    ulong p = x[0].ncol;
    auto grad = zerosColumn!T(p);
    T X2 = 0;
    for(ulong i = 0; i < nBlocks; ++i)
    {
      n += y[i].length;
      auto tmpGrad = gradient_(distrib, link, y[i], x[i], mu[i], eta[i]);
      grad += tmpGrad.grad;
      X2 += tmpGrad.X2;
    }
    auto df = cast(T)(n - p);
    T phi = X2/df;
    assert(df > 0, "Number of items n is not greater than the number of parameters p.");
    grad /= phi;
    return dotSum!(T)(grad, dir);
  }
  T dPhi(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, T alpha, ColumnVector!T dir, 
            ColumnVector!(T) _coef, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!T offset)
  {
    auto coef = _coef.dup;
    coef += alpha*dir;
    ulong nBlocks = y.length;
    ulong p = x[0].ncol;

    auto eta = new ColumnVector!(T)[nBlocks];

    foreach(i; taskPool.parallel(iota(nBlocks)))
      eta[i] = mult_(x[i], coef);
    
    if(offset.length != 0)
    {
      foreach(i; taskPool.parallel(iota(nBlocks)))
        eta[i] += offset[i];
    }
    auto mu = link.linkinv(dataType, eta);
    
    auto nStore = taskPool.workerLocalStorage(cast(ulong)0);
    auto gradStore = taskPool.workerLocalStorage(zerosColumn!T(p));
    auto X2Store = taskPool.workerLocalStorage(cast(T)0);
    T X2 = 0;
    
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      nStore.get += y[i].length;
      auto tmp = gradient_(distrib, link, y[i], x[i], mu[i], eta[i]);
      gradStore.get += tmp.grad;
      X2Store.get += tmp.X2;
    }
    
    ulong n = 0;
    auto grad = zerosColumn!T(p);
    foreach(_n; nStore.toRange)
      n += _n;
    foreach(_grad; gradStore.toRange)
      grad += _grad;
    foreach(_X2; X2Store.toRange)
      X2 += _X2;
    
    auto df = cast(T)(n - p);
    T phi = X2/df;
    
    assert(df > 0, "Number of items n is not greater than the number of parameters p.");
    grad /= phi;
    return dotSum!(T)(grad, dir);
  }
}
/*
  Numerical Optimization, 2nd Edition J. Nocedal & S. J. Wright p58-p59
*/
class Interpolation(T)
{
  public:
  /* Quadratic interpolation in the interval [0, alpha0] */
  T quadratic(T alpha0, T phi0, T dphi0, T phiAlpha0)
  {
    return -(dphi0 * (alpha0^^2))/(2*(phiAlpha0 - phi0 - dphi0*alpha0));
  }
  /* Cubic interpolation in the interval [alpha0, alpha1] */
  T cubic(T alpha0, T alpha1, T phi0, T phi1, T dphi0, T dphi1)
  {
    T d1 = dphi0 + dphi1 - 3*((phi0 - phi1)/(alpha0 - alpha1));
    T d2 = sgn(alpha1 - alpha0)*((d1^^2 - dphi0*dphi1)^^0.5);
    T alpha = alpha1 - (alpha1 - alpha0)*(
                        (dphi1 + d2 - d1)/(dphi1 - dphi0 + 2*d2));
    return alpha;
  }
}

/*
  Objects for calculating the gradient
*/
class Gradient(T, CBLAS_LAYOUT layout = CblasColMajor)
if(isFloatingPoint!T)
{
  private:
  auto gradient_(AbstractDistribution!T distrib, 
      AbstractLink!T link, ColumnVector!T y, Matrix!(T, layout) x, 
      ColumnVector!T mu, ColumnVector!T eta)
  {
    ulong p = x.ncol;
    ulong ni = x.nrow;
    auto grad = zerosColumn!T(p);
    auto tmp = zerosColumn!(T)(ni);
    T numer = 0;
    T X2 = 0;
    for(ulong i = 0; i < ni; ++i)
    {
      numer = y[i] - mu[i];
      X2 += (numer^^2)/distrib.variance(mu[i]);
      tmp[i] = numer/(link.deta_dmu(mu[i], eta[i]) * distrib.variance(mu[i]));
      for(ulong j = 0; j < p; ++j)
      {
        grad[j] += tmp[i] * x[i, j];
      }
    }
    return tuple!("X2", "grad")(X2, grad);
  }

  public:
  ColumnVector!(T) gradient(RegularData dataType, AbstractDistribution!T distrib, 
              AbstractLink!T link, /* ColumnVector!T dir, T alpha, */
              ColumnVector!(T) coef, ColumnVector!T y, 
              Matrix!(T, layout) x, ColumnVector!T offset)
  {
    /* coef += alpha*dir; */
    auto eta = mult_(x, coef);
    
    if(offset.length != 0)
      eta += offset;
    
    auto mu = link.linkinv(eta);

    ulong p = x.ncol; ulong n = x.nrow;
    auto tmpGrad = gradient_(distrib, link, y, x, mu, eta);
    auto df = cast(T)(n - p);

    assert(df > 0, "Number of items n is not greater than the number of parameters p.");
    T phi = tmpGrad.X2/df;
    return tmpGrad.grad/phi;
  }
  ColumnVector!(T) gradient(Block1D dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, /* ColumnVector!T dir, T alpha, */
            ColumnVector!(T) coef, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!T offset)
  {
    /* coef += alpha*dir; */
    ulong nBlocks = y.length; ulong n = 0;
    auto eta = new ColumnVector!(T)[nBlocks];

    for(ulong i = 0; i < nBlocks; ++i)
      eta[i] = mult_(x[i], coef);
    
    if(offset.length != 0)
    {
      for(ulong i = 0; i < nBlocks; ++i)
        eta[i] += offset[i];
    }
    auto mu = link.linkinv(eta);

    ulong p = x[0].ncol;
    auto grad = zerosColumn!T(p);
    T X2 = 0;
    for(ulong i = 0; i < nBlocks; ++i)
    {
      n += y[i].length;
      auto tmpGrad = gradient_(distrib, link, y[i], x[i], mu[i], eta[i]);
      grad += tmpGrad.grad;
      X2 += tmpGrad.X2;
    }
    auto df = cast(T)(n - p);
    assert(df > 0, "Number of items n is not greater than the number of parameters p.");
    T phi = X2/df;
    return grad/phi;
  }
  ColumnVector!(T) gradient(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, /* ColumnVector!T dir, T alpha, */
            ColumnVector!(T) coef, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!T offset)
  {
    /* coef += alpha*dir; */
    ulong nBlocks = y.length;
    ulong p = x[0].ncol;

    auto eta = new ColumnVector!(T)[nBlocks];

    foreach(i; taskPool.parallel(iota(nBlocks)))
      eta[i] = mult_(x[i], coef);
    
    if(offset.length != 0)
    {
      foreach(i; taskPool.parallel(iota(nBlocks)))
        eta[i] += offset[i];
    }
    auto mu = link.linkinv(dataType, eta);
    
    auto nStore = taskPool.workerLocalStorage(cast(ulong)0);
    auto gradStore = taskPool.workerLocalStorage(zerosColumn!T(p));
    auto X2Store = taskPool.workerLocalStorage(cast(T)0);
    T X2 = 0;
    
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      nStore.get += y[i].length;
      auto tmp = gradient_(distrib, link, y[i], x[i], mu[i], eta[i]);
      gradStore.get += tmp.grad;
      X2Store.get += tmp.X2;
    }
    
    ulong n = 0;
    auto grad = zerosColumn!T(p);
    foreach(_n; nStore.toRange)
      n += _n;
    foreach(_grad; gradStore.toRange)
      grad += _grad;
    foreach(_X2; X2Store.toRange)
      X2 += _X2;
    
    auto df = cast(T)(n - p);
    
    assert(df > 0, "Number of items n is not greater than the number of parameters p.");
    T phi = X2/df;

    return grad/phi;
  }
}

/*
  Design for line search algorithms
*/
interface AbstractLineSearch(T, CBLAS_LAYOUT layout = CblasColMajor)
{
  public:
  T linesearch(RegularData dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, 
               ColumnVector!(T) coef, ColumnVector!T y, 
               Matrix!(T, layout) x, ColumnVector!T weights, 
               ColumnVector!T offset);
  
  T linesearch(Block1D dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, 
               ColumnVector!(T) coef, BlockColumnVector!T y, 
               BlockMatrix!(T, layout) x, BlockColumnVector!T weights, 
               BlockColumnVector!T offset);
  
  T linesearch(Block1DParallel dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, 
               ColumnVector!(T) coef, BlockColumnVector!T y, 
               BlockMatrix!(T, layout) x, BlockColumnVector!T weights, 
               BlockColumnVector!T offset);
}

/*
  Backtracking Line Search
*/
class BackTrackingLineSearch(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractLineSearch!(T, layout)
if(isFloatingPoint!T)
{
  private:
  T alpha0; /* Initial Step length */
  immutable(T) c; /* Wolfe condition */
  T rho; /* Contraction factor */
  ulong maxit; /* Maximum number of iterations */
  ulong exitCode; /* Exit code for line search, -1 unfinished, 0 successful completion, 1 maximum limit reached */
  
  public:
  /*
    The body of these line search functions are the same, needs to be 
    refactored.
  */
  T linesearch(RegularData dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, 
               ColumnVector!(T) coef, ColumnVector!T y, Matrix!(T, layout) x,
               ColumnVector!T weights = zerosColumn!(T)(0), 
               ColumnVector!T offset = zerosColumn!(T)(0))
  {
    T alpha = alpha0;
    auto f = new Deviance!(T, layout)();
    auto dev = f.deviance(dataType, distrib, link, coef, y, x, 
                          weights, offset);
    auto df = new Gradient!(T, layout);
    ulong iter = 0;
    while(true)
    {
      auto coefNew = coef + alpha*dir;
      auto devNew = f.deviance(dataType, distrib, link, coefNew, y, x, 
                          weights, offset);
      auto deriv = df.gradient(dataType, distrib, link, coef, y, x, offset);
      if(devNew <= dev + c*alpha*dotSum!(T)(deriv, dir))
        break;
      alpha *= rho;
      ++iter;
      if(iter >= maxit)
      {
        writeln("Maximum number of line search iterations exceeded.");
        exitCode = 1;
        return alpha;
      }
    }
    exitCode = 0;
    return alpha;
  }
  T linesearch(Block1D dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, ColumnVector!(T) coef,
               BlockColumnVector!T y, BlockMatrix!(T, layout) x, 
               BlockColumnVector!T weights = new ColumnVector!(T)[0], 
               BlockColumnVector!T offset = new ColumnVector!(T)[0])
  {
    T alpha = alpha0;
    auto f = new Deviance!(T, layout)();
    auto dev = f.deviance(dataType, distrib, link, coef, y, x, 
                          weights, offset);
    auto df = new Gradient!(T, layout);
    ulong iter = 0;
    while(true)
    {
      auto coefNew = coef + alpha*dir;
      auto devNew = f.deviance(dataType, distrib, link, coefNew, y, x, 
                          weights, offset);
      auto deriv = df.gradient(dataType, distrib, link, coef, y, x, offset);
      if(devNew <= dev + c*alpha*dotSum!(T)(deriv, dir))
        break;
      alpha *= rho;
      ++iter;
      if(iter >= maxit)
      {
        writeln("Maximum number of line search iterations exceeded.");
        exitCode = 1;
        return alpha;
      }
    }
    exitCode = 0;
    return alpha;
  }
  T linesearch(Block1DParallel dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, 
               ColumnVector!(T) coef, BlockColumnVector!T y, 
               BlockMatrix!(T, layout) x,
               BlockColumnVector!T weights = new ColumnVector!(T)[0], 
               BlockColumnVector!T offset = new ColumnVector!(T)[0])
  {
    T alpha = alpha0;
    auto f = new Deviance!(T, layout)();
    auto dev = f.deviance(dataType, distrib, link, coef, y, x, 
                          weights, offset);
    auto df = new Gradient!(T, layout);
    ulong iter = 0;
    while(true)
    {
      auto coefNew = coef + alpha*dir;
      auto devNew = f.deviance(dataType, distrib, link, coefNew, y, x, 
                          weights, offset);
      auto deriv = df.gradient(dataType, distrib, link, coef, y, x, offset);
      if(devNew <= dev + c*alpha*dotSum!(T)(deriv, dir))
        break;
      alpha *= rho;
      ++iter;
      if(iter >= maxit)
      {
        writeln("Maximum number of line search iterations exceeded.");
        exitCode = 1;
        return alpha;
      }
    }
    exitCode = 0;
    return alpha;
  }
  this(T _alpha = 1, T _c = 1E-4, T _rho = 0.5, ulong _maxit = 25)
  {
    alpha0 = _alpha; c = _c;
    rho = _rho; maxit = _maxit;
    exitCode = -1;
  }
}


/*
  Implementation of LBFGS Solver

  Reference:  Numerical Optimization 2nd Edition, 
              J. Nocedal, S. J. Wright.
*/
class LBFGSSolver(T, CBLAS_LAYOUT layout = CblasColMajor)
{
  public:
  ulong m; /* Number of recent {sk, yk} pairs to keep */
  ulong idx; /* Number of valid vectors so far */
  ulong iter; /* Iteration */
  ColumnVector!(T) rho;
  Matrix!(T, layout) sm; /* Each column denotes the respective vector sk */
  Matrix!(T, layout) ym; /* Each column denotes the respective vector yk */
  
  this(ulong p, ulong _m)
  {
    m = _m; idx = 0; rho = new ColumnVector!(T)(m);
    iter = 1; sm = zerosMatrix!(T, layout)(p, m);
    ym = zerosMatrix!(T, layout)(p, m);
  }
  ColumnVector!(T) solve(RegularData dataType, 
               AbstractLineSearch!(T, layout) linesearch,
               AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!(T) _coef, 
               ColumnVector!T y, Matrix!(T, layout) x, 
               ColumnVector!T weights = zerosColumn!(T)(0), 
               ColumnVector!T offset = zerosColumn!(T)(0))
  {
    auto sk = cast(ColumnVector!(T))sm.columnSelect(idx);
    auto yk = cast(ColumnVector!(T))ym.columnSelect(idx);
    auto H0 = dotSum!(T)(sk, yk)/dotSum!(T)(yk, yk);
    writeln("H0: ", H0);
    /**************** Algorithm 7.4 Start ****************/
    auto dF = new Gradient!(double)();
    /*
      You may want to include coef inside this solver in the future?
    */
    auto coef = _coef.dup;
    auto coefOld = _coef.dup;
    auto grad = dF.gradient(dataType, distrib, link,
                coef, y, x, offset);
    writeln("Gradient Calculated: ", grad.array);
    auto q = grad.dup;
    auto alphai = new ColumnVector!(T)(m);
    writeln("Stop Point 1");
    for(ulong i = 0; i < (idx + 1); ++i)
    {
      sk = cast(ColumnVector!(T))sm.columnSelect(i);
      yk = cast(ColumnVector!(T))ym.columnSelect(i);
      alphai[i] = rho[i]*dotSum!(T)(sk, q);
      q -= alphai[i]*yk;
    }
    writeln("Stop Point 2");
    auto pk = H0*q;
    for(ulong i = idx; i > 0; --i)
    {
      writeln("i: ", i, ", idx: ", idx);
      sk = cast(ColumnVector!(T))sm.columnSelect(i);
      yk = cast(ColumnVector!(T))ym.columnSelect(i);
      auto beta = rho[i]*dotSum!(T)(yk, pk);
      pk = pk + sk*(alphai[i] - beta);
    }
    writeln("Stop Point 3");
    /**************** Algorithm 7.4 End ****************/

    /* Line Search */
    auto alpha = linesearch.linesearch(dataType, distrib, 
                    link, pk, coef, y, x, weights, offset);
    writeln("Alpha: ", alpha);
    coef += alpha*pk;
    idx = min(idx + 1, m - 1);

    auto gradOld = grad.dup;
    grad = dF.gradient(dataType, distrib, link,
                coef, y, x, offset);
    
    if(iter > m)
    {
      /* Remove the first column */
      sm.refColumnRemove(0);
      sk = coef - coefOld;
      yk = grad - gradOld;
      /* Append new columns */
      ym.appendColumn(yk);
      sm.appendColumn(sk);
    }else{
      sk = coef - coefOld;
      yk = grad - gradOld;
      ym.refColumnAssign(yk, idx);
      sm.refColumnAssign(sk, idx);
    }

    iter += 1;
    
    return coef;
  }
  
  //ColumnVector!(T) solve(Block1D dataType, 
  //             AbstractLineSearch!(T, layout) linesearch,
  //             AbstractDistribution!T distrib, 
  //             AbstractLink!T link, ColumnVector!(T) _coef, 
  //             BlockColumnVector!T y, BlockMatrix!(T, layout) x, 
  //             BlockColumnVector!T weights = new ColumnVector!(double)[0], 
  //             BlockColumnVector!T offset = new ColumnVector!(double)[0]);
  //
  //ColumnVector!(T) solve(Block1DParallel dataType, 
  //             AbstractLineSearch!(T, layout) linesearch,
  //             AbstractDistribution!T distrib, 
  //             AbstractLink!T link, ColumnVector!(T) _coef, 
  //             BlockColumnVector!T y, BlockMatrix!(T, layout) x, 
  //             BlockColumnVector!T weights = new ColumnVector!(double)[0], 
  //             BlockColumnVector!T offset = new ColumnVector!(double)[0]);
}

