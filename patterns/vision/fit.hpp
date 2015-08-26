#ifndef __FIT_FPP__
#define __FIT_FPP__
#include "core.hpp"
#include "stats.hpp"

template<typename T, typename F = T>
F parabolicFit ( const T left, const T center, const T right, T *maxp = NULL)
{
    const bool cl (center == left);
    const bool cr (center == right);
	
    if ((!(center > left || cl) && !(center > right || cr)) &&
        (!(center < left || cl) && !(center < right || cr)))
        return std::numeric_limits<F>::infinity();

	
    const F s = left + right;
    const F d = right - left;
    const F c = center + center;
	
    // case: center is at max
    if (c == s)
    {
        if (maxp) *maxp = center;
        return 0;
    }
	
    // ymax for the quadratic
    if (maxp) *maxp = (4 * c * c - 4 * c * s + d * d) / (8 * (c - s));
	
    // position of the max
    return d / (2 * (c - s));
	
}


// 1D correlation: Normalized Correlation
//
template <class Iterator>
double normalized1DCorr (Iterator Ib, Iterator Ie, Iterator Mb)
{
  double sii(0.0), si(0.0), smm(0.0), sm(0.0), sim (0.0);
  int n;
  double rsq;

  n = Ie - Ib;
  assert (n);

  // Get container value type
  typedef typename std::iterator_traits<Iterator>::value_type value_type;

  Iterator ip = Ib;
  Iterator mp = Mb;

  while (ip < Ie)
    {
      double iv (*ip);
      double mv (*mp);
      si += iv;
      sm += mv;
      sii += (iv * iv);
      smm += (mv * mv);
      sim += (iv * mv);

      ip++;
      mp++;
    }

  double cross = ((n * sim) - (si * sm));
  double energyA = ((n * sii) - (si * si));
  double energyB = ((n * smm) - (sm * sm));

  energyA = energyA * energyB;
  rsq = 0.;
  if (energyA != 0.)
    rsq = (cross * cross) / energyA;

  return rsq;
}


template<> 
inline float parabolicFit (const float left, const float center, const float right, float *maxp)
{
	const bool cl (core::RealEq (center - left, (float) (1.e-10)));
	const bool cr (core::RealEq (center - right, (float) (1.e-10)));
	
	if ((!(center > left || cl ) && !(center > right || cr )) &&
			(!(center < left || cl ) && !(center < right || cr )))
            return std::numeric_limits<float>::infinity();
	
	
	const float s = left + right;
	const float d = right - left;
	const float c = center + center;
	
	// case: center is at max
	if (c == s)
    {
			if (maxp) *maxp = center;
			return 0;
    }
	
	// ymax for the quadratic
	if (maxp) *maxp = (4 * c * c - 4 * c * s + d * d) / (8 * (c - s));
	
	// position of the max
	return d / (2 * (c - s));
	
}



#endif
