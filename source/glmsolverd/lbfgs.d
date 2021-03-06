module glmsolverd.lbfgs;

import std.conv: to;
import std.stdio: writeln;
import std.typecons: Tuple, tuple;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import std.parallelism;
import std.range: iota;

import std.math: abs, isNaN, pow, sgn;
import std.algorithm: min, max;
import std.algorithm.iteration: mean;

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
    auto regData = new RegularData();
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(regData, distrib, link, mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) W(Block1DParallel dataType, 
              AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    auto regData = new RegularData();
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = W(regData, distrib, link, mu[i], eta[i]);
    return ret;
  }
}

/* XWX Class */
class XWX(T, CBLAS_LAYOUT layout = CblasColMajor)
if(isFloatingPoint!T)
{
  public:
  void XWX(RegularData dataType, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
  }
  void XWX(Block1D dataType, ref Matrix!(T, layout) xwx, 
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
  ColumnVector!(T) etaRegular;
  ColumnVector!(T) muRegular;
  ColumnVector!(T)[] etaBlock;
  ColumnVector!(T)[] muBlock;
  ColumnVector!(T) gradient(RegularData dataType, AbstractDistribution!T distrib, 
              AbstractLink!T link, /* ColumnVector!T dir, T alpha, */
              ColumnVector!(T) coef, ColumnVector!T y, 
              Matrix!(T, layout) x, ColumnVector!T offset)
  {
    /* coef += alpha*dir; */
    auto eta = mult_(x, coef);
    
    if(offset.length != 0)
      eta += offset;
    
    etaRegular = eta;
    auto mu = link.linkinv(eta);
    muRegular = mu;

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
    etaBlock = eta;
    auto mu = link.linkinv(eta);
    muBlock = mu;

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
    etaBlock = eta;
    auto mu = link.linkinv(dataType, eta);
    muBlock = mu;
    
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
  void setAlpha0(T alpha);
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
  immutable(T) rho; /* Contraction factor */
  immutable(ulong) maxit; /* Maximum number of iterations */
  ulong exitCode; /* Exit code for line search, -1 unfinished, 0 successful completion, 1 maximum limit reached */
  
  public:
  this(T _alpha = 1, T _c = 1E-4, T _rho = 0.5, ulong _maxit = 25)
  {
    alpha0 = _alpha; c = _c;
    rho = _rho; maxit = _maxit;
    exitCode = -1;
  }
  void setAlpha0(T alpha)
  {
    alpha0 = alpha;
  }
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
      
      //writeln("Debug print from line search, dev: ", dev, ", devNew: ", devNew);

      if(devNew <= dev + c*alpha*dotSum!(T)(deriv, dir))
        break;
      
      alpha *= rho;
      ++iter;
      
      if(iter >= maxit)
      {
        writeln("Maximum number of line search iterations reached.");
        exitCode = 1;
        return alpha;
      }
    }
    //writeln("Evaluations of line search loop: ", iter);
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
        writeln("Maximum number of line search iterations reached.");
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
        writeln("Maximum number of line search iterations reached.");
        exitCode = 1;
        return alpha;
      }
    }
    exitCode = 0;
    return alpha;
  }
}

class WolfeLineSearch(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractLineSearch!(T, layout)
{
  private:
  T alphaMax;
  T alphaStart;
  immutable(T) c1;
  immutable(T) c2;
  immutable(long) maxit;
  long exitCode;
  
  public:
  this(T _alpha = 1, T _alphaMax = 5, T _c1 = 1E-4, 
          T _c2 = 0.9, long _maxit = 25)
  {
    alphaMax = _alphaMax;
    c1 = _c1; c2 = _c2; maxit = _maxit;
    exitCode = -1; alphaStart = _alpha;
  }
  void setAlpha0(T alpha)
  {
    alphaStart = alpha;
  }
  
  T zoom(RegularData dataType, T alphaLo, T alphaHi, T phi0, T dphi0, 
        AbstractDistribution!T distrib, AbstractLink!T link, 
        ColumnVector!T dir, ColumnVector!(T) coef, 
        ColumnVector!T y, Matrix!(T, layout) x, 
        ColumnVector!T weights = zerosColumn!(T)(0), 
        ColumnVector!T offset = zerosColumn!(T)(0))
  {
    assert(alphaLo < alphaHi, "alphaLo: " ~ to!(string)(alphaLo) ~ " is not less than alphaHi: " ~ to!(string)(alphaHi) ~ ".");
    auto interp = new Interpolation!(T)();
    auto F = new Deviance!(T, layout)();
    auto dF = new Gradient!(T, layout);
    long iter = 1;
    while(true)
    {
      auto delta = alphaLo*dir;
      auto coefLo = coef + delta;
      auto phiLo = F.deviance(dataType, distrib, link, coefLo, y, x, 
                      weights, offset);
      auto gradLo = dF.gradient(dataType, distrib, link, coefLo, 
                      y, x, offset);
      auto dphiLo = dotSum!(T)(gradLo, dir);
      
      delta = alphaLo*dir;
      auto coefHi = coef + delta;
      auto phiHi = F.deviance(dataType, distrib, link, coefHi, y, x, 
                      weights, offset);
      auto gradHi = dF.gradient(dataType, distrib, link,
                            coefHi, y, x, offset);
      auto dphiHi = dotSum!(T)(gradHi, dir);
      
      writeln("alphaLo: ", alphaLo, ", alphaHi: ", alphaHi);
      T alpha = interp.cubic(alphaLo, alphaHi, phiLo, phiHi, dphiLo, dphiHi);
      writeln("zoom() iteration: ", iter, ", Current alpha: ", alpha);
      delta = alpha*dir;
      auto coefNew = coef + delta;
      T phiNew = F.deviance(dataType, distrib, link, coefNew, 
                      y, x, weights, offset);
      if((phiNew > phi0 + c1*alpha*dphi0) | (phiNew >= phiLo))
      {
        writeln("From zoom(): failed wolfe condition: phiNew: ", phiNew, ", phiCheck: ", phi0+ c1*alpha*dphi0, ", phiLo: ", phiLo);
        alphaHi = alpha;
      }else{
        auto gradNew = dF.gradient(dataType, distrib, link,
                            coefNew, y, x, offset);
        T dphiNew = dotSum!(T)(gradNew, dir);
        if(abs(dphiNew) <= -c2*dphi0)
        {
          exitCode = 0;
          return alpha;
        }
        if(dphiNew*(alphaHi - alphaLo) >= 0)
        {
          alphaHi = alphaLo;
        }
        alphaLo = alpha;
      }
      if(iter >= maxit)
      {
        writeln("Maximum number of zoom iterations reached.");
        exitCode = 1;
        return alpha;
      }
      ++iter;
    }
  }

  T linesearch(RegularData dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, 
               ColumnVector!(T) coef, ColumnVector!T y, Matrix!(T, layout) x,
               ColumnVector!T weights = zerosColumn!(T)(0), 
               ColumnVector!T offset = zerosColumn!(T)(0))
  {
    auto F = new Deviance!(T, layout)();
    auto dF = new Gradient!(T, layout);
    long iter = 1;
    /* alpha_{i - 1}, alpha_{i}, alpha_{i + 1} */
    T[2] alpha = new T[2]; T[2] phi = new T[2];
    alpha[0] = 0; alpha[1] = alphaStart;
    //writeln("alphaStart: ", alpha[1]);
    auto phi0 = F.deviance(dataType, distrib, link, coef, y, x, 
                          weights, offset);
    auto grad0 = dF.gradient(dataType, distrib, link, coef, y, x, offset);
    auto dphi0 = dotSum!(T)(grad0, dir);
    writeln("coef: ", coef.array);
    while(true)
    {
      writeln("Line search iteration: ", iter);
      auto delta = alpha[1]*dir;
      writeln("dir: ", dir.array);
      auto coefi = coef + delta;
      auto tmp = F.deviance(dataType, distrib, link, coefi, 
                      y, x, weights, offset);
      phi[0] = phi[1]; phi[1] = tmp;
      writeln("alpha at top of loop: ", alpha);
      writeln("phi at top of loop: ", phi);
      writeln("phi0 at top of loop: ", phi0);
      writeln("dphi0 at top of loop: ", dphi0);
      if((phi[1] > phi0 + c1*alpha[1]*dphi0) | ((iter > 1) && (phi[1] >= phi[0])))
      {
        writeln("zoom() call 1");
        exitCode = 0;
        alpha[1] = zoom(dataType, alpha[0], alpha[1], phi0, dphi0, distrib, 
                      link, dir, coef, y, x, weights, offset);
        writeln("alpha exit: ", alpha[1]);
        return alpha[1];
      }
      auto gradi = dF.gradient(dataType, distrib, link, coefi, y, x, offset);
      auto dphii = dotSum!(T)(gradi, dir);
      if(abs(dphii) <= -c2*dphi0)
      {
        exitCode = 0;
        return alpha[1];
      }
      if(dphii >= 0)
      {
        writeln("zoom() call 2, dphii: ", dphii);
        /* delete this hack */
        if(alpha[1] > alpha[0])
        {
          writeln("Line search hack swap");
          auto tmp0 = alpha[0];
          alpha[0] = alpha[1];
          alpha[1] = tmp0;
        }
        exitCode = 0;
        alpha[1] = zoom(dataType, alpha[1], alpha[0], phi0, dphi0, distrib, 
                      link, dir, coef, y, x, weights, offset);
        writeln("alpha exit: ", alpha[1]);
        return alpha[1];
      }
      alpha[0] = alpha[1];
      alpha[1] = alpha[1] + 0.5*(alphaMax - alpha[1]);
      if(iter >= maxit)
      {
        writeln("Maximum number of line search iterations reached, alpha: ", alpha[1]);
        exitCode = 1;
        return alpha[1];
      }
      ++iter;
    }
  }

  T linesearch(Block1D dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, 
               ColumnVector!(T) coef, BlockColumnVector!T y, 
               BlockMatrix!(T, layout) x, BlockColumnVector!T weights, 
               BlockColumnVector!T offset)
  {
    return 0;
  }
  
  T linesearch(Block1DParallel dataType, AbstractDistribution!T distrib, 
               AbstractLink!T link, ColumnVector!T dir, 
               ColumnVector!(T) coef, BlockColumnVector!T y, 
               BlockMatrix!(T, layout) x, BlockColumnVector!T weights, 
               BlockColumnVector!T offset)
  {
    return 0;
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
  immutable(ulong) m; /* Number of recent {sk, yk} pairs to keep */
  ulong idx; /* Number of valid vectors so far */
  ulong iter; /* Iteration */
  immutable(T) epsilon;
  bool calcAlpha0; /* Whether alpha0 should be calculated or just set to 1 at each line search iteration */
  ColumnVector!(T) coef;
  ColumnVector!(T) coefold;
  ColumnVector!(T) rho;
  ColumnVector!(T) pk;
  Matrix!(T, layout) sm; /* Each column denotes the respective vector sk */
  Matrix!(T, layout) ym; /* Each column denotes the respective vector yk */
  
  this(ulong p, ulong _m, T _epsilon = 1E-5, bool _calcAlpha0 = false)
  {
    m = _m; idx = 0; rho = new ColumnVector!(T)(m);
    iter = 1; sm = zerosMatrix!(T, layout)(p, m);
    ym = zerosMatrix!(T, layout)(p, m);
    coef = zerosColumn!(T)(p); coefold = zerosColumn!(T)(p);
    epsilon = _epsilon; calcAlpha0 = _calcAlpha0;
  }
  this(ColumnVector!(T) _coef, ulong p, ulong _m, T _epsilon = 1E-5, bool _calcAlpha0 = false)
  {
    m = _m; idx = 0; rho = new ColumnVector!(T)(m);
    iter = 1; sm = zerosMatrix!(T, layout)(p, m);
    ym = zerosMatrix!(T, layout)(p, m);
    coef = _coef.dup; coefold = _coef.dup;
    epsilon = _epsilon; calcAlpha0 = _calcAlpha0;
  }
  /* Refactor this solve function */
  ColumnVector!(T) solve(RegularData dataType, 
               AbstractLineSearch!(T, layout) linesearch,
               AbstractDistribution!T distrib, 
               AbstractLink!T link, /* ColumnVector!(T) _coef, */
               ColumnVector!T y, Matrix!(T, layout) x, 
               ColumnVector!T weights = zerosColumn!(T)(0), 
               ColumnVector!T offset = zerosColumn!(T)(0))
  {
    //writeln("Iteration: ", iter);
    //writeln("idx: ", idx);
    //writeln("Check S: ", sm);
    //writeln("Check Y: ", ym);
    auto sk = cast(ColumnVector!(T))sm.columnSelect(idx);
    auto yk = cast(ColumnVector!(T))ym.columnSelect(idx);
    auto H0 = dotSum!(T)(sk, yk)/dotSum!(T)(yk, yk);
    //writeln("H0: ", H0);
    /**************** Algorithm 7.4 Start ****************/
    auto dF = new Gradient!(double)();

    //auto coef = _coef.dup;
    //auto coefold = _coef.dup;
    auto grad = dF.gradient(dataType, distrib, link,
                coef, y, x, offset);
    //writeln("Algo 7.4 gradient: ", grad.array);
    auto q = grad.dup;
    auto alphai = new ColumnVector!(T)(m);
    
    for(long i = 0; i < (idx + 1); ++i)
    {
      sk = cast(ColumnVector!(T))sm.columnSelect(i);
      yk = cast(ColumnVector!(T))ym.columnSelect(i);
      //writeln("i: ", i, ", sk: ", sk.array);
      //writeln("i: ", i, ", yk: ", yk.array);
      alphai[i] = rho[i]*dotSum!(T)(sk, q);
      q -= alphai[i]*yk;
    }
    //writeln("q: ", q.array);

    pk = H0*q;
    for(long i = idx; i >= 0; --i)
    {
      //writeln("i: ", i);
      sk = cast(ColumnVector!(T))sm.columnSelect(i);
      yk = cast(ColumnVector!(T))ym.columnSelect(i);
      //writeln("i: ", i, ", sk: ", sk.array);
      //writeln("i: ", i, ", yk: ", yk.array);
      auto beta = rho[i]*dotSum!(T)(yk, pk);
      pk = pk + sk*(alphai[i] - beta);
    }
    //writeln("Output direction (pk): ", pk.array);
    /**************** Algorithm 7.4 End ****************/
    pk *= -1;

    /* Line Search */
    auto alpha = linesearch.linesearch(dataType, distrib, 
                    link, pk, coef, y, x, weights, offset);
    //writeln("Alpha: ", alpha);
    coefold = coef.dup;
    coef += alpha*pk;
    idx = min(idx + 1, m - 1);

    auto gradold = grad.dup;
    grad = dF.gradient(dataType, distrib, link,
                coef, y, x, offset);
    
    //writeln("coef: ", coef.array);
    //writeln("coefold: ", coefold.array);
    //writeln("grad: ", grad.array);
    //writeln("gradold: ", gradold.array);
    if(iter >= m)
    {
      sk = coef - coefold;
      yk = grad - gradold;
      rho[idx] = 1/dotSum!(T)(yk, sk);
      /* Removing first and Append new columns */
      sm.refColumnRemove(0);
      ym.refColumnRemove(0);
      sm.appendColumn(sk);
      ym.appendColumn(yk);
      //writeln("Updated S: ", sm);
      //writeln("Updated Y: ", ym);
    }else{
      sk = coef - coefold;
      yk = grad - gradold;
      rho[idx] = 1/dotSum!(T)(yk, sk);
      ym.refColumnAssign(yk, idx);
      sm.refColumnAssign(sk, idx);
    }

    iter += 1;
    
    return coef;
  }
  
  ColumnVector!(T) solve(Block1D dataType, 
               AbstractLineSearch!(T, layout) linesearch,
               AbstractDistribution!T distrib, 
               AbstractLink!T link, /* ColumnVector!(T) _coef, */
               BlockColumnVector!T y, BlockMatrix!(T, layout) x, 
               BlockColumnVector!T weights, 
               BlockColumnVector!T offset)
  {
    auto sk = cast(ColumnVector!(T))sm.columnSelect(idx);
    auto yk = cast(ColumnVector!(T))ym.columnSelect(idx);
    auto H0 = dotSum!(T)(sk, yk)/dotSum!(T)(yk, yk);
    /**************** Algorithm 7.4 Start ****************/
    auto dF = new Gradient!(double)();
    auto grad = dF.gradient(dataType, distrib, link,
                coef, y, x, offset);
    auto q = grad.dup;
    auto alphai = new ColumnVector!(T)(m);
    
    for(long i = 0; i < (idx + 1); ++i)
    {
      sk = cast(ColumnVector!(T))sm.columnSelect(i);
      yk = cast(ColumnVector!(T))ym.columnSelect(i);
      alphai[i] = rho[i]*dotSum!(T)(sk, q);
      q -= alphai[i]*yk;
    }

    pk = H0*q;
    for(long i = idx; i >= 0; --i)
    {
      sk = cast(ColumnVector!(T))sm.columnSelect(i);
      yk = cast(ColumnVector!(T))ym.columnSelect(i);
      auto beta = rho[i]*dotSum!(T)(yk, pk);
      pk = pk + sk*(alphai[i] - beta);
    }
    /**************** Algorithm 7.4 End ****************/
    pk *= -1;

    /* Line Search */
    auto alpha = linesearch.linesearch(dataType, distrib, 
                    link, pk, coef, y, x, weights, offset);
    coefold = coef.dup;
    coef += alpha*pk;
    idx = min(idx + 1, m - 1);

    auto gradold = grad.dup;
    grad = dF.gradient(dataType, distrib, link,
                coef, y, x, offset);
    
    if(iter >= m)
    {
      sk = coef - coefold;
      yk = grad - gradold;
      rho[idx] = 1/dotSum!(T)(yk, sk);
      sm.refColumnRemove(0);
      ym.refColumnRemove(0);
      sm.appendColumn(sk);
      ym.appendColumn(yk);
    }else{
      sk = coef - coefold;
      yk = grad - gradold;
      rho[idx] = 1/dotSum!(T)(yk, sk);
      ym.refColumnAssign(yk, idx);
      sm.refColumnAssign(sk, idx);
    }

    iter += 1;
    
    return coef;
  }
  
  ColumnVector!(T) solve(Block1DParallel dataType, 
               AbstractLineSearch!(T, layout) linesearch,
               AbstractDistribution!T distrib, 
               AbstractLink!T link, /* ColumnVector!(T) _coef, */
               BlockColumnVector!T y, BlockMatrix!(T, layout) x, 
               BlockColumnVector!T weights, 
               BlockColumnVector!T offset)
  {
    auto sk = cast(ColumnVector!(T))sm.columnSelect(idx);
    auto yk = cast(ColumnVector!(T))ym.columnSelect(idx);
    auto H0 = dotSum!(T)(sk, yk)/dotSum!(T)(yk, yk);
    /**************** Algorithm 7.4 Start ****************/
    auto dF = new Gradient!(double)();
    auto grad = dF.gradient(dataType, distrib, link,
                coef, y, x, offset);
    auto q = grad.dup;
    auto alphai = new ColumnVector!(T)(m);
    
    for(long i = 0; i < (idx + 1); ++i)
    {
      sk = cast(ColumnVector!(T))sm.columnSelect(i);
      yk = cast(ColumnVector!(T))ym.columnSelect(i);
      alphai[i] = rho[i]*dotSum!(T)(sk, q);
      q -= alphai[i]*yk;
    }

    pk = H0*q;
    for(long i = idx; i >= 0; --i)
    {
      sk = cast(ColumnVector!(T))sm.columnSelect(i);
      yk = cast(ColumnVector!(T))ym.columnSelect(i);
      auto beta = rho[i]*dotSum!(T)(yk, pk);
      pk = pk + sk*(alphai[i] - beta);
    }
    /**************** Algorithm 7.4 End ****************/
    pk *= -1;

    /* Line Search */
    auto alpha = linesearch.linesearch(dataType, distrib, 
                    link, pk, coef, y, x, weights, offset);
    coefold = coef.dup;
    coef += alpha*pk;
    idx = min(idx + 1, m - 1);

    auto gradold = grad.dup;
    grad = dF.gradient(dataType, distrib, link,
                coef, y, x, offset);
    
    if(iter >= m)
    {
      sk = coef - coefold;
      yk = grad - gradold;
      rho[idx] = 1/dotSum!(T)(yk, sk);
      sm.refColumnRemove(0);
      ym.refColumnRemove(0);
      sm.appendColumn(sk);
      ym.appendColumn(yk);
    }else{
      sk = coef - coefold;
      yk = grad - gradold;
      rho[idx] = 1/dotSum!(T)(yk, sk);
      ym.refColumnAssign(yk, idx);
      sm.refColumnAssign(sk, idx);
    }

    iter += 1;
    
    return coef;
  }
}

/* GLM Functions */
/***********************************************************************/
/**
  GLM function using LBFGS with Regular Data
*/
auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(
       RegularData dataType, Matrix!(T, layout) x, 
       Matrix!(T, layout) _y, AbstractDistribution!(T) distrib,
       AbstractLink!(T) link, LBFGSSolver!(T, layout) solver,
       AbstractLineSearch!(T, layout) linesearch,
       AbstractInverse!(T, layout) inverse = new GETRIInverse!(T, layout)(),
       Control!T control = new Control!T(),
       bool calculateCovariance = true, 
       ColumnVector!(T) offset = zerosColumn!(T)(0),
       ColumnVector!(T) weights = zerosColumn!(T)(0))
if(isFloatingPoint!(T))
{
  auto init = distrib.init(new GradientDescent!(T, layout)(1), _y, weights);
  auto y = init[0]; weights = init[1];
  
  // Initialize with link function
  //solver.coef[0] = mean(link.linkfun(y).array);
  //writeln("Initial coefficients: ", solver.coef.array);
  
  bool converged, badBreak, doOffset, doWeights;
  converged = true;
  
  if(offset.len != 0)
    doOffset = true;
  if(weights.len != 0)
    doWeights = true;
  
  auto gradient = new Gradient!(double)();
  auto grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
  auto gradold = grad.dup;
  
  auto deviance = new Deviance!(T, layout)();
  auto dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);
  auto devold = dev;
  
  /* Gradient descent with line search for first iteration */
  auto alphaInit = linesearch.linesearch(dataType, distrib, 
                    link, grad, solver.coef, y, x, weights, 
                    offset);
  //writeln("Alpha init: ", alphaInit);
  solver.coefold = solver.coef.dup;
  auto dir = alphaInit*grad;
  solver.coef += dir;
  
  dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);
  grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
  
  writeln("GLM Dev (first): ", dev, ", devold: ", devold);
  
  auto sk = solver.coef - solver.coefold;
  auto yk = grad - gradold;
  
  solver.sm.refColumnAssign(sk, 0);
  solver.ym.refColumnAssign(yk, 0);
  solver.rho[0] = 1/dotSum!(T)(yk, sk);
  
  T alpha0;
  if(solver.calcAlpha0)
  {
    alpha0 = 2*(dev - devold)/dotSum!(T)(grad, dir);
    alpha0 = alpha0 > 0 ? alpha0 : 1; /* Sanity check */
    alpha0 = min(1, alpha0*1.01);
    //writeln("The first alpha0: ", alpha0);
  }else{
    alpha0 = 1;
  }
  
  linesearch.setAlpha0(alpha0);
  solver.solve(dataType, linesearch, distrib, link, 
            y, x, weights, offset);
  
  auto absErr = T.infinity;
  auto relErr = T.infinity;
  auto gradErr = T.infinity;
  
  devold = dev;
  dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);
  
  writeln("GLM Dev (second): ", dev, ", devold: ", devold);

  gradold = grad.dup;
  grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
  //writeln("Refreshed old gradient: ", gradold.array);
  //writeln("Refreshed gradient: ", grad.array);

  absErr = absoluteError(dev, devold);
  relErr = relativeError(dev, devold);
  gradErr = norm(grad);//relativeError(grad, gradold);
  
  while(true)
  {
    //writeln("Relative error: ", relErr);
    //writeln("Gradient norm: ", gradErr);
    if((relErr < control.epsilon) & /* (gradErr < solver.epsilon) & */ (solver.iter >= solver.m))
      break;
    
    if(solver.calcAlpha0)
    {
      alpha0 = 2*(dev - devold)/dotSum!(T)(grad, solver.pk);
      alpha0 = alpha0 > 0 ? alpha0 : 1; /* Sanity check */
      alpha0 = min(1, alpha0*1.01);
      //writeln("Check alpha0: ", alpha0);
    }else{
      alpha0 = 1;
    }
        
    linesearch.setAlpha0(alpha0);
    solver.solve(dataType, linesearch, distrib, link, 
              y, x, weights, offset);
    
    devold = dev;
    dev = deviance.deviance(dataType, distrib, link, 
                    solver.coef, y, x, weights, offset);
    
    writeln("GLM iteration: ", solver.iter, ", dev: ", dev, ", devold: ", devold);
    
    gradold = grad.dup;
    grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
    
    absErr = absoluteError(dev, devold);
    relErr = relativeError(dev, devold);
    gradErr = norm(grad);//relativeError(grad, gradold);
    
    if(control.maxit < solver.iter)
    {
      converged = false;
      break;
    }
  }

  T phi = 1; long p = x.ncol; long n = x.nrow;
  Matrix!(T, layout) cov, xw, xwx, R;
  if(calculateCovariance)
  {
    auto mu = gradient.muRegular;
    auto eta = gradient.etaRegular;
    auto z = link.Z(y, mu, eta);
    
    if(doOffset)
      z = map!( (x1, x2) => x1 - x2 )(z, offset);
    
    auto W = new Weights!(T, layout)();
    /* Weights calculation standard vs sqrt */
    auto w = W.W(dataType, distrib, link, mu, eta);
    
    if(doWeights)
      w = map!( (x1, x2) => x1*x2 )(w, weights);
    
    //writeln("Coefficients: ", coef.array);
    
    auto XWX = new XWX!(T, layout)();
    XWX.XWX(dataType, xwx, xw, x, z, w);
    
    cov = inverse.inv(xwx);
    
    if(!unitDispsersion!(T, typeof(distrib)))
    {
      phi = dev/(n - p);
      imap!( (T x) => x*phi)(cov);
    }
  }else{
    cov = fillMatrix!(T)(1, p, p);
  }
  auto obj = new GLM!(T, layout)(solver.iter, converged, phi, distrib, link, solver.coef, cov, dev, absErr, relErr);
  return obj;
}


/**
  GLM function using LBFGS with Block1D Data
*/
auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(
       Block1D dataType, Matrix!(T, layout)[] x, 
       Matrix!(T, layout)[] _y, AbstractDistribution!(T) distrib,
       AbstractLink!(T) link, LBFGSSolver!(T, layout) solver,
       AbstractLineSearch!(T, layout) linesearch,
       AbstractInverse!(T, layout) inverse = new GETRIInverse!(T, layout)(),
       Control!T control = new Control!T(),
       bool calculateCovariance = true, 
       ColumnVector!(T)[] offset = new ColumnVector!(T)[0],
       ColumnVector!(T)[] weights = new ColumnVector!(T)[0])
if(isFloatingPoint!(T))
{
  auto init = distrib.init(new GradientDescent!(T, layout)(1), _y, weights);
  auto y = init[0]; weights = init[1];
  auto nBlocks = y.length;

  long p = x[0].ncol;
  long n = 0;
  for(long i = 0; i < nBlocks; ++i)
    n += x[i].nrow;
  
  // Initialize with link function
  //solver.coef[0] = mean(link.linkfun(y).array);
  //writeln("Initial coefficients: ", solver.coef.array);
  
  bool converged, badBreak, doOffset, doWeights;
  converged = true;
  
  if(offset.length != 0)
    doOffset = true;
  if(weights.length != 0)
    doWeights = true;
  
  auto gradient = new Gradient!(double)();
  auto grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
  auto gradold = grad.dup;
  
  auto deviance = new Deviance!(T, layout)();
  auto dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);
  auto devold = dev;
  
  /* Gradient descent with line search for first iteration */
  auto alphaInit = linesearch.linesearch(dataType, distrib, 
                    link, grad, solver.coef, y, x, weights, 
                    offset);
  
  solver.coefold = solver.coef.dup;
  auto dir = alphaInit*grad;
  solver.coef += dir;
  
  dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);
  grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
  
  auto sk = solver.coef - solver.coefold;
  auto yk = grad - gradold;
  
  solver.sm.refColumnAssign(sk, 0);
  solver.ym.refColumnAssign(yk, 0);
  solver.rho[0] = 1/dotSum!(T)(yk, sk);
  
  T alpha0;
  if(solver.calcAlpha0)
  {
    alpha0 = 2*(dev - devold)/dotSum!(T)(grad, dir);
    alpha0 = alpha0 > 0 ? alpha0 : 1; /* Sanity check */
    alpha0 = min(1, alpha0*1.01);
    
  }else{
    alpha0 = 1;
  }
  
  linesearch.setAlpha0(alpha0);
  solver.solve(dataType, linesearch, distrib, link, 
            y, x, weights, offset);
  
  auto absErr = T.infinity;
  auto relErr = T.infinity;
  auto gradErr = T.infinity;
  
  devold = dev;
  dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);

  gradold = grad.dup;
  grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);

  absErr = absoluteError(dev, devold);
  relErr = relativeError(dev, devold);
  gradErr = norm(grad);
  
  while(true)
  {
    if((relErr < control.epsilon) & /* (gradErr < solver.epsilon) & */ (solver.iter >= solver.m))
      break;
    
    if(solver.calcAlpha0)
    {
      alpha0 = 2*(dev - devold)/dotSum!(T)(grad, solver.pk);
      alpha0 = alpha0 > 0 ? alpha0 : 1; /* Sanity check */
      alpha0 = min(1, alpha0*1.01);
    }else{
      alpha0 = 1;
    }
        
    linesearch.setAlpha0(alpha0);
    solver.solve(dataType, linesearch, distrib, link, 
              y, x, weights, offset);
    
    devold = dev;
    dev = deviance.deviance(dataType, distrib, link, 
                    solver.coef, y, x, weights, offset);
    
    gradold = grad.dup;
    grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
    
    absErr = absoluteError(dev, devold);
    relErr = relativeError(dev, devold);
    gradErr = norm(grad);
    
    if(control.maxit < solver.iter)
    {
      converged = false;
      break;
    }
  }

  T phi = 1;
  Matrix!(T, layout) cov, xwx, R;
  Matrix!(T, layout)[] xw;
  if(calculateCovariance)
  {
    auto mu = gradient.muBlock;
    auto eta = gradient.etaBlock;
    auto z = Z!(T)(link, y, mu, eta);
    if(doOffset)
    {
      for(ulong i = 0; i < nBlocks; ++i)
        z[i] = map!( (x1, x2) => x1 - x2 )(z[i], offset[i]);
    }
    
    auto W = new Weights!(T, layout)();
    auto w = W.W(dataType, distrib, link, mu, eta);
    
    if(doWeights)
    {
      for(ulong i = 0; i < nBlocks; ++i)
        w[i] = map!( (x1, x2) => x1*x2 )(w[i], weights[i]);
    }
    
    auto XWX = new XWX!(T, layout)();
    XWX.XWX(dataType, xwx, xw, x, z, w);
    
    cov = inverse.inv(xwx);
    
    if(!unitDispsersion!(T, typeof(distrib)))
    {
      phi = dev/(n - p);
      imap!( (T x) => x*phi)(cov);
    }
  }else{
    cov = fillMatrix!(T)(1, p, p);
  }
  auto obj = new GLM!(T, layout)(solver.iter, converged, phi, distrib, link, solver.coef, cov, dev, absErr, relErr);
  return obj;
}


/**
  GLM function using LBFGS with Block1D Data
*/
auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(
       Block1DParallel dataType, Matrix!(T, layout)[] x, 
       Matrix!(T, layout)[] _y, AbstractDistribution!(T) distrib,
       AbstractLink!(T) link, LBFGSSolver!(T, layout) solver,
       AbstractLineSearch!(T, layout) linesearch,
       AbstractInverse!(T, layout) inverse = new GETRIInverse!(T, layout)(),
       Control!T control = new Control!T(),
       bool calculateCovariance = true, 
       ColumnVector!(T)[] offset = new ColumnVector!(T)[0],
       ColumnVector!(T)[] weights = new ColumnVector!(T)[0])
if(isFloatingPoint!(T))
{
  auto init = distrib.init(new GradientDescent!(T, layout)(1), _y, weights);
  auto y = init[0]; weights = init[1];
  auto nBlocks = y.length;

  long p = x[0].ncol;
  long n = 0;
  auto nStore = taskPool.workerLocalStorage(0L);
  /* Parallelised reduction required */
  foreach(i; taskPool.parallel(iota(nBlocks)))
    nStore.get += x[i].nrow;
  foreach (_n; nStore.toRange)
        n += _n;
  
  // Initialize with link function
  //solver.coef[0] = mean(link.linkfun(y).array);
  //writeln("Initial coefficients: ", solver.coef.array);
  
  bool converged, badBreak, doOffset, doWeights;
  converged = true;
  
  if(offset.length != 0)
    doOffset = true;
  if(weights.length != 0)
    doWeights = true;
  
  auto gradient = new Gradient!(double)();
  auto grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
  auto gradold = grad.dup;
  
  auto deviance = new Deviance!(T, layout)();
  auto dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);
  auto devold = dev;
  
  /* Gradient descent with line search for first iteration */
  auto alphaInit = linesearch.linesearch(dataType, distrib, 
                    link, grad, solver.coef, y, x, weights, 
                    offset);
  
  solver.coefold = solver.coef.dup;
  auto dir = alphaInit*grad;
  solver.coef += dir;
  
  dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);
  grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
  
  auto sk = solver.coef - solver.coefold;
  auto yk = grad - gradold;
  
  solver.sm.refColumnAssign(sk, 0);
  solver.ym.refColumnAssign(yk, 0);
  solver.rho[0] = 1/dotSum!(T)(yk, sk);
  
  T alpha0;
  if(solver.calcAlpha0)
  {
    alpha0 = 2*(dev - devold)/dotSum!(T)(grad, dir);
    alpha0 = alpha0 > 0 ? alpha0 : 1; /* Sanity check */
    alpha0 = min(1, alpha0*1.01);
    
  }else{
    alpha0 = 1;
  }
  
  linesearch.setAlpha0(alpha0);
  solver.solve(dataType, linesearch, distrib, link, 
            y, x, weights, offset);
  
  auto absErr = T.infinity;
  auto relErr = T.infinity;
  auto gradErr = T.infinity;
  
  devold = dev;
  dev = deviance.deviance(dataType, distrib, link, 
                  solver.coef, y, x, weights, offset);

  gradold = grad.dup;
  grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);

  absErr = absoluteError(dev, devold);
  relErr = relativeError(dev, devold);
  gradErr = norm(grad);
  
  while(true)
  {
    if((relErr < control.epsilon) & /* (gradErr < solver.epsilon) & */ (solver.iter >= solver.m))
      break;
    
    if(solver.calcAlpha0)
    {
      alpha0 = 2*(dev - devold)/dotSum!(T)(grad, solver.pk);
      alpha0 = alpha0 > 0 ? alpha0 : 1; /* Sanity check */
      alpha0 = min(1, alpha0*1.01);
    }else{
      alpha0 = 1;
    }
        
    linesearch.setAlpha0(alpha0);
    solver.solve(dataType, linesearch, distrib, link, 
              y, x, weights, offset);
    
    devold = dev;
    dev = deviance.deviance(dataType, distrib, link, 
                    solver.coef, y, x, weights, offset);
    
    gradold = grad.dup;
    grad = gradient.gradient(dataType, distrib, link,
                solver.coef, y, x, offset);
    
    absErr = absoluteError(dev, devold);
    relErr = relativeError(dev, devold);
    gradErr = norm(grad);
    
    if(control.maxit < solver.iter)
    {
      converged = false;
      break;
    }
  }

  T phi = 1;
  Matrix!(T, layout) cov, xwx, R;
  Matrix!(T, layout)[] xw;
  if(calculateCovariance)
  {
    auto mu = gradient.muBlock;
    auto eta = gradient.etaBlock;
    auto z = Z!(T)(link, y, mu, eta);
    if(doOffset)
    {
      foreach(i; taskPool.parallel(iota(nBlocks)))
          z[i] = map!( (x1, x2) => x1 - x2 )(z[i], offset[i]);
    }
    
    auto W = new Weights!(T, layout)();
    auto w = W.W(dataType, distrib, link, mu, eta);
    
    if(doWeights)
    {
      foreach(i; taskPool.parallel(iota(nBlocks)))
        w[i] = map!( (x1, x2) => x1*x2 )(w[i], weights[i]);
    }
    
    auto XWX = new XWX!(T, layout)();
    XWX.XWX(dataType, xwx, xw, x, z, w);
    
    cov = inverse.inv(xwx);
    
    if(!unitDispsersion!(T, typeof(distrib)))
    {
      phi = dev/(n - p);
      imap!( (T x) => x*phi)(cov);
    }
  }else{
    cov = fillMatrix!(T)(1, p, p);
  }
  auto obj = new GLM!(T, layout)(solver.iter, converged, phi, distrib, link, solver.coef, cov, dev, absErr, relErr);
  return obj;
}
