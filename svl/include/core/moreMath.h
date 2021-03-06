

#ifndef  _MATH_H_
#define  _MATH_H_
#include <stddef.h>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace svl {

    
    double exponentialMovingAverageIrregular(
                                             double alpha, double sample, double prevSample,
                                             double deltaTime, double emaPrev );
    double exponentialMovingAverage( double sample, double alpha );
    
    /*
     * Exponential Smoothing
     */
    
    template<typename T = double>
    class eMAvg{
    public:
        eMAvg(const T alpha,  const T start_value) : m_alpha(alpha), m_value(start_value) {}
        
        bool is_valid () const { return m_alpha >= 0.0 && m_alpha <= 1.0; }
        
        // reworking next = (new_val * alpha) + ((1-alpha) * current) to produce incremental change
        T update(const T new_val) const {
            return (m_value -= m_alpha * (m_value - new_val));
        }
        
        const T& current () const { return m_value; }
        const T& alpah () const { return m_alpha; }
    private:
        mutable T m_alpha;//variable (in [0,1])
        mutable T m_value;//current value
        
    };
    
    /*
     * Quick rolling window 1d median of size 3
     */
#ifndef median_of_3
    
#define median_of_3(a, b, c)                                       \
((a) > (b) ? ((a) < (c) ? (a) : ((b) < (c) ? (c) : (b))) :    \
((a) > (c) ? (a) : ((b) < (c) ? (b) : (c))))
    
#endif
    
template <typename Iter, typename T = typename std::iterator_traits<Iter>::value_type>
static bool rolling_median_3 (Iter begin, Iter endd, std::vector<T>& dst)
{
    dst.push_back(*begin++);
    Iter one = begin;
    Iter last = endd - 1;
    for (auto ctr = one; ctr < last; ctr++){
        T m = *(ctr-1);
        T c = *(ctr);
        T p = *(ctr+1);
        T val = median_of_3(m,c,p);
        dst.push_back(val);
    }
    dst.push_back(*last);
    return true;
}

/*
 Simple 1D filter interface and median filtering derivation
 
 */

template <class T>
class FilterInterface
{
    
public:
    virtual std::vector<T> filter(const std::vector<T> & in) = 0;
    
private:
    
    
};

template <class T>
class median1D : public FilterInterface<T>
{
    std::vector<T> _history;
    std::vector<T> _pool;
    unsigned       _median;
    
public:
    
    median1D(int32_t window_size)
    :
    _history(keep_odd(window_size), T()),
    _pool(_history),
    _median(window_size / 2 + 1)
    {
        assert(window_size >= 3);
    }
    
    
    std::vector<T> filter(const std::vector<T> & in)
    {
        assert(in.size() > 0);
        unsigned hist_ptr = 0;
        std::fill(_history.begin(), _history.end(), in[0]);
        std::vector<T> out;
        out.reserve(in.size());
        for(auto x : in)
        {
            _history[hist_ptr] = x;
            hist_ptr = (hist_ptr + 1) % _history.size();
            std::copy(_history.begin(), _history.end(), _pool.begin());
            auto first = _pool.begin();
            auto last  = _pool.end();
            auto middle = first + (last - first) / 2;
            
            // Sort from first to last around the middle to get the median
            std::nth_element(first, middle, last);
            out.push_back(*middle);
        }
        return out;
    }
    
    unsigned keep_odd(unsigned n)
    {
        return (n % 2 == 0) ? (n + 1) : n;
    }
};
    
    
template <class T> class samples_1D;

    
    //---------------------------------------------------------------------------------------------------------------------------
    //-------------- lin_reg_results: a simple helper class for dealing with output from linear regression routines -------------
    //---------------------------------------------------------------------------------------------------------------------------
    //This routine is a helper class for holding the results from either weighted or un-weighted linear regression routines.
    // It mostly holds data as members, but also has some routines for using the results to interpolate and pretty-printing.
    //
    // NOTE: All parameters should be inspected for infinity and NaN before using in any calculations. Not all members are
    //       filled by each routine.
    //
    // NOTE: Not all data is always used for all routines. The weighted regression routines will omit some values (e.g., NaNs)
    //       and possibly unweighted (i.e., infinitely surpressed) values. Always check N or dof instead of ...size().
    //
    // NOTE: For more info the parameters here, look at the regression routine notes. They are detailed in context there.
    //
    template <class T> class lin_reg_results {
    public:
        // ===================================== Fit parameters ========================================
        T slope             = std::numeric_limits<T>::quiet_NaN(); //m in y=m*x+b.
        T sigma_slope       = std::numeric_limits<T>::quiet_NaN(); //Uncertainty in m.
        T intercept         = std::numeric_limits<T>::quiet_NaN(); //b in y=m*x+b.
        T sigma_intercept   = std::numeric_limits<T>::quiet_NaN(); //Uncertainty in b.
        
        // ================= Quantities related to the fit and/or distribution of data =================
        T N                 = std::numeric_limits<T>::quiet_NaN(); //# of datum used (not the DOF, int casted to a T).
        T dof               = std::numeric_limits<T>::quiet_NaN(); //# of degrees of freedom used.
        T sigma_f           = std::numeric_limits<T>::quiet_NaN(); //Std. dev. of the data.
        T covariance        = std::numeric_limits<T>::quiet_NaN(); //Covariance.
        T lin_corr          = std::numeric_limits<T>::quiet_NaN(); //Linear corr. coeff. (aka "r", "Pearson's coeff").
        
        // -------------------------------- un-weighted regression only --------------------------------
        T sum_sq_res        = std::numeric_limits<T>::quiet_NaN(); //Sum-of-squared residuals.
        T tvalue            = std::numeric_limits<T>::quiet_NaN(); //t-value for r.
        T pvalue            = std::numeric_limits<T>::quiet_NaN(); //Two-tailed p-value for r.
        
        // --------------------------------- weighted regression only ----------------------------------
        T chi_square        = std::numeric_limits<T>::quiet_NaN(); //Sum-of-squared residuals weighted by sigma_f_i^{-2}.
        T qvalue            = std::numeric_limits<T>::quiet_NaN(); //q-value for chi-square.
        T cov_params        = std::numeric_limits<T>::quiet_NaN(); //Covariance of the slope and intercept.
        T corr_params       = std::numeric_limits<T>::quiet_NaN(); //Correlation of sigma_slope and sigma_intercept.
        
        // ========== Quantities which might be of interest for computing other quantities =============
        
        // -------------------------------- un-weighted regression only --------------------------------
        T mean_x            = std::numeric_limits<T>::quiet_NaN(); //Mean of all x_i for used datum.
        T mean_f            = std::numeric_limits<T>::quiet_NaN(); //Mean of all f_i for used datum.
        T sum_x             = std::numeric_limits<T>::quiet_NaN(); //Sum of x_i.
        T sum_f             = std::numeric_limits<T>::quiet_NaN(); //Sum of f_i.
        T sum_xx            = std::numeric_limits<T>::quiet_NaN(); //Sum of x_i*x_i.
        T sum_xf            = std::numeric_limits<T>::quiet_NaN(); //Sum of f_i*f_i.
        T Sxf               = std::numeric_limits<T>::quiet_NaN(); //Sum of (x_i - mean_x)*(f_i - mean_f).
        T Sxx               = std::numeric_limits<T>::quiet_NaN(); //Sum of (x_i - mean_x)^2.
        T Sff               = std::numeric_limits<T>::quiet_NaN(); //Sum of (f_i - mean_f)^2.
        
        // --------------------------------- weighted regression only ----------------------------------
        // NOTE: sigma_f'_i is an equivalent sigma_f_i which incorporates sigma_f_i and sigma_x_i using a rough guess
        //       of the slope.
        T mean_wx           = std::numeric_limits<T>::quiet_NaN(); //Mean of all x_i/(sigma_f'_i^2) for used datum.
        T mean_wf           = std::numeric_limits<T>::quiet_NaN(); //Mean of all f_i/(sigma_f'_i^2) for used datum.
        
        T sum_w             = std::numeric_limits<T>::quiet_NaN(); //Sum of 1/(sigma_f'_i^2).
        T sum_wx            = std::numeric_limits<T>::quiet_NaN(); //Sum of x_i/(sigma_f'_i^2).
        T sum_wf            = std::numeric_limits<T>::quiet_NaN(); //Sum of f_i/(sigma_f'_i^2).
        T sum_wxx           = std::numeric_limits<T>::quiet_NaN(); //Sum of x_i*x_i/(sigma_f'_i^2).
        T sum_wxf           = std::numeric_limits<T>::quiet_NaN(); //Sum of f_i*f_i/(sigma_f'_i^2).
        
        // ==================================== Member functions =======================================
        lin_reg_results();
        
        //Evaluates the model at the given point. Ignores uncertainties.
        T evaluate_simple(T x) const;
        
        //Sample the line over the given range at 'n' equally-spaced points. Ignores uncertainties.
        samples_1D<T> sample_uniformly_over(T xmin, T xmax, size_t n = 5) const;
        
        //Aligns into a simple table.
        std::string display_table(void) const;
        
        //Compare all members which aren't NAN. Aligns into a simple table.
        std::string comparison_table(const lin_reg_results<T> &in) const;
    };

    
//---------------------------------------------------------------------------------------------------------------------------
//---------------------------------------- vec_3: A three-dimensional vector -------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------
//This class is the basis of the geometry-heavy classes below. It can happily be used on its own.
template <class T> class vec_3 {
    public:
        using value_type = T;
        T x, y, z;
    
        //Constructors.
        vec_3();
        vec_3(T a, T b, T c);
        vec_3( const vec_3 & );
    
        //Operators - vec_3 typed.
        vec_3 & operator=(const vec_3 &);
    
        vec_3   operator+(const vec_3 &) const;
        vec_3 & operator+=(const vec_3 &);
        vec_3   operator-(const vec_3 &) const;
        vec_3 & operator-=(const vec_3 &);
    
        bool operator==(const vec_3 &) const;
        bool operator!=(const vec_3 &) const;
        bool operator<(const vec_3 &) const;   //This is mostly for sorting / placing into a std::map. There is no great
                                              // way to define '<' for a vector, so be careful with it.
        bool isfinite(void) const;        //Logical AND of std::isfinite() on each coordinate.

        vec_3   operator*(const T &) const;
        vec_3 & operator*=(const T &);
        vec_3   operator/(const T &) const;
        vec_3 & operator/=(const T &);

        //Operators - T typed.
        T & operator[](size_t);
    
        //Member functions.
        T Dot(const vec_3 &) const;        // ---> Dot product.
        vec_3 Cross(const vec_3 &) const;   // ---> Cross product.
        vec_3 Mask(const vec_3 &) const;    // ---> Term-by-term product:   (a b c).Outer(1 2 0) = (a 2*b 0)
    
        vec_3 unit() const;                // ---> Return a normalized version of this vector.
        T length() const;                 // ---> (pythagorean) length of vector.
        T distance(const vec_3 &) const;   // ---> (pythagorean) distance between vectors.
        T sq_dist(const vec_3 &) const;    // ---> Square of the (pythagorean) distance. Avoids a sqrt().
        T angle(const vec_3 &, bool *OK=nullptr) const;  // ---> The |angle| (in radians, [0:pi]) separating two vectors.

        vec_3 zero(void) const; //Returns a zero-vector.

        vec_3<T> rotate_around_x(T angle_rad) const; // Rotate by some angle (in radians, [0:pi]) around a cardinal axis.
        vec_3<T> rotate_around_y(T angle_rad) const;
        vec_3<T> rotate_around_z(T angle_rad) const;

        bool GramSchmidt_orthogonalize(vec_3<T> &, vec_3<T> &) const; //Using *this as seed, orthogonalize (n.b. not orthonormalize) the inputs.

        std::string to_string(void) const;
       // vec_3<T> from_string(const std::string &in); //Sets *this and returns a copy.
    
        //Friends.
        template<class Y> friend std::ostream & operator << (std::ostream &, const vec_3<Y> &); // ---> Overloaded stream operators.
     //   template<class Y> friend std::istream & operator >> (std::istream &, vec_3<Y> &);
};


//This is a function for rotation unit vectors in some plane. It requires angles to describe the plane of rotation, angle of rotation. 
// It also requires a unit vector with which to rotate the plane about.
//
// Note: Prefer Gram-Schmidt orthogonalization or the cross-product if you can. This routine has some unfortunate poles...
vec_3<double> rotate_unit_vector_in_plane(const vec_3<double> &A, const double &theta, const double &R);

//This function evolves a pair of position and velocity (x(t=0),v(t=0)) to a pair (x(t=T),v(t=T)) using the
// classical expression for a time- and position-dependent force F(x;t). It is highly unstable, so the number
// of iterations must be specified. If this is going to be used for anything important, make sure that the
// number of iterations is chosen sufficiently high so as to produce negligible errors.
std::tuple<vec_3<double>,vec_3<double>> Evolve_x_v_over_T_via_F(const std::tuple<vec_3<double>,vec_3<double>> &x_and_v,
                                                              std::function<vec_3<double>(vec_3<double> x, double T)> F,
                                                              double T, long int steps);


//---------------------------------------------------------------------------------------------------------------------------
//----------------------------------------- vec_2: A two-dimensional vector --------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------
//This class is the basis of the geometry-heavy classes below. It can happily be used on its own.
template <class T> class vec_2 {
    public:
        using value_type = T;
        T x, y;

        //Constructors.
        vec_2();
        vec_2(T a, T b);
        vec_2( const vec_2 & );

        //Operators - vec_2 typed.
        vec_2 & operator=(const vec_2 &);

        vec_2   operator+(const vec_2 &) const;
        vec_2 & operator+=(const vec_2 &);
        vec_2   operator-(const vec_2 &) const;
        vec_2 & operator-=(const vec_2 &);

        bool operator==(const vec_2 &) const;
        bool operator!=(const vec_2 &) const;
        bool operator<(const vec_2 &) const;   //This is mostly for sorting / placing into a std::map. There is no great
                                              // way to define '<' for a vector, so be careful with it.
        bool isfinite(void) const;        //Logical AND of std::isfinite() on each coordinate.

        vec_2   operator*(const T &) const;
        vec_2 & operator*=(const T &);
        vec_2   operator/(const T &) const;
        vec_2 & operator/=(const T &);

        //Operators - T typed.
        T & operator[](size_t);

        //Member functions.
        T Dot(const vec_2 &) const;        // ---> Dot product.
        vec_2 Mask(const vec_2 &) const;    // ---> Term-by-term product:   (a b).Outer(5 0) = (5*a 0*b)

        vec_2 unit() const;                // ---> Return a normalized version of this vector.
        T length() const;                 // ---> (pythagorean) length of vector.
        T distance(const vec_2 &) const;   // ---> (pythagorean) distance between vectors.
        T sq_dist(const vec_2 &) const;    // ---> Square of the (pythagorean) distance. Avoids a sqrt().

        vec_2 zero(void) const; //Returns a zero-vector.

        vec_2<T> rotate_around_z(T angle_rad) const; // Rotate by some angle (in radians, [0:pi]) around a cardinal axis.

        std::string to_string(void) const;
     //   vec_2<T> from_string(const std::string &in); //Sets *this and returns a copy.

        //Friends.
        template<class Y> friend std::ostream & operator << (std::ostream &, const vec_2<Y> &); // ---> Overloaded stream operators.
//        template<class Y> friend std::istream & operator >> (std::istream &, vec_2<Y> &);
};

    template <class T>   class line {
    public:
        using value_type = T;
        vec_3<T> R_0;  //A point which the line intersects.
        vec_3<T> U_0;  //A unit vector which points along the length of the line.
        
        //Constructors.
        line();
        line(const vec_3<T> &R_A, const vec_3<T> &R_B);
        
        //Member functions.
        T Distance_To_Point( const vec_3<T> & ) const;
        
        bool Intersects_With_Line_Once( const line<T> &, vec_3<T> &) const;
        
        bool Closest_Point_To_Line( const line<T> &, vec_3<T> &) const;
        
        vec_3<T> Project_Point_Orthogonally( const vec_3<T> & ) const; // Projects the given point onto the nearest point on the line.
        
        //Friends.
        template<class Y> friend std::ostream & operator << (std::ostream &, const line<Y> &); // ---> Overloaded stream operators.
        template<class Y> friend std::istream & operator >> (std::istream &, line<Y> &);
    };
    

    
template <class T> class samples_1D {
    public:
        //--------------------------------------------------- Data members -------------------------------------------------
        //Samples are laid out like [x_i, sigma_x_i, f_i, sigma_f_i] sequentially, so if there are three samples data will be
        // laid out in memory like:
        //   [x_0, sigma_x_0, f_0, sigma_f_0, x_1, sigma_x_1, f_1, sigma_f_1, x_2, sigma_x_2, f_2, sigma_f_2 ].
        // and can thus be used as a c-style array if needed. If you can pass stride and offset (e.g., to GNU GSL), use:
        //   - for x_i:        stride = 4,  offset from this->samples.data() to first element = 0. 
        //   - for sigma_x_i:  stride = 4,  offset from this->samples.data() to first element = 1. 
        //   - for f_i:        stride = 4,  offset from this->samples.data() to first element = 2. 
        //   - for sigma_f_i:  stride = 4,  offset from this->samples.data() to first element = 3. 
        std::vector<std::array<T,4>> samples;

        //This flag controls how uncertainties are treated. By default, no assumptions about normality are made. If you are
        // certain that the sample uncertainties are: independent, random, and normally-distributed (i.e., no systematic 
        // biases are thought to be present, and the underlying measurement process follows a normal distribution) then this
        // flag can be set to reduce undertainties in accordance with the knowledge that arithmetical combinations of values 
        // tend to cancel uncertainties. Be CERTAIN to actually inspect this fact!
        //
        // Conversely, setting this flag may under- or over-value uncertainties because we completely ignore covariance.
        // If it is false, you at least know for certain that you are over-valuing uncertainties. Be aware that BOTH 
        // settings imply that the uncertainties are considered small enough that a linear error propagation procedure 
        // remains approximately valid!
        bool uncertainties_known_to_be_independent_and_random = false;        


        //User-defined metadata, useful for packing along extra information about the data. Be aware that the data can
        // become stale if you are not updating it as needed!
        std::map<std::string,std::string> metadata;


        //-------------------------------------------------- Constructors --------------------------------------------------
        samples_1D();
        samples_1D(const samples_1D<T> &in);
        samples_1D(std::vector<std::array<T,4>> samps);

        //Providing [x_i, f_i] data. Assumes sigma_x_i and sigma_f_i uncertainties are (T)(0).
        samples_1D(const std::list<vec_2<T>> &samps);

        //------------------------------------------------ Member Functions ------------------------------------------------
        samples_1D<T> operator=(const samples_1D<T> &rhs);

        //If not supplied, sigma_x_i and sigma_f_i uncertainties are assumed to be (T)(0).
        void push_back(T x_i, T sigma_x_i, T f_i, T sigma_f_i, bool inhibit_sort = false);
        void push_back(const std::array<T,4> &samp, bool inhibit_sort = false);
        void push_back(const std::array<T,2> &x_dx, const std::array<T,2> &y_dy, bool inhibit_sort = false);
        void push_back(T x_i, T f_i, bool inhibit_sort = false);
        void push_back(const vec_2<T> &x_i_and_f_i, bool inhibit_sort = false);
        void push_back(T x_i, T f_i, T sigma_f_i, bool inhibit_sort = false);

        bool empty(void) const;  // == this->samples.empty()
        size_t size(void) const; // == this->samples.size()

        //An explicit way for the user to sort. Not needed unless you are directly editing this->samples for some reason.
        void stable_sort(void); //Sorts on x-axis. Lowest-first.

        //Get the datum with the minimum and maximum x_i or f_i. If duplicates are found, there is no rule specifying which.
        std::pair<std::array<T,4>,std::array<T,4>> Get_Extreme_Datum_x(void) const;
        std::pair<std::array<T,4>,std::array<T,4>> Get_Extreme_Datum_y(void) const;

        //Normalizes data so that \int_{-inf}^{inf} f(x) (times) f(x) dx = 1 multiplying by constant factor.
        void Normalize_wrt_Self_Overlap(void);

        //Sets uncertainties to zero. Useful in certain situations, such as computing aggregate std dev's.
        samples_1D<T> Strip_Uncertainties_in_x(void) const;
        samples_1D<T> Strip_Uncertainties_in_y(void) const;

        //Ensure there is a single datum with the given x_i (within +-eps), averaging coincident data if necessary.
        void Average_Coincident_Data(T eps);

        //Replaces {x,y}-values with rank. {dup,N}-plicates get an averaged (maybe non-integer) rank.
        samples_1D<T> Rank_x(void) const;
        samples_1D<T> Rank_y(void) const;

        //Swaps x_i with f_i and sigma_x_i with sigma_f_i.
        samples_1D<T> Swap_x_and_y(void) const;
        
        //Sum of all x_i or f_i. Uncertainties are propagated properly.
        std::array<T,2> Sum_x(void) const;
        std::array<T,2> Sum_y(void) const;

        //Computes the [statistical mean (average), std dev of the *data*]. Be careful to choose the MEAN or AVERAGE 
        // carefully -- the uncertainty is different! See source for more info.
        std::array<T,2> Average_x(void) const; 
        std::array<T,2> Average_y(void) const;

        //Computes the [statistical mean (average), std dev of the *mean*]. Be careful to choose the MEAN or AVERAGE 
        // carefully -- the uncertainty is different! See source for more info.
        std::array<T,2> Mean_x(void) const;
        std::array<T,2> Mean_y(void) const;

        //Computes the [statistical weighted mean (average), std dev of the *weighted mean*]. See source. 
        std::array<T,2> Weighted_Mean_x(void) const;
        std::array<T,2> Weighted_Mean_y(void) const;

        //Finds or computes the median datum with linear interpolation halfway between datum if N is odd.
        std::array<T,2> Median_x(void) const;
        std::array<T,2> Median_y(void) const;

        //Select those within [L,H] (inclusive). Ignores sigma_x_i for inclusivity purposes. Keeps uncertainties intact.
        samples_1D<T> Select_Those_Within_Inc(T L, T H) const;

        //Returns [at_x, sigma_at_x (==0), f, sigma_f] interpolated at at_x. If at_x is not within the range of the 
        // samples, expect {at_x,(T)(0),(T)(0),(T)(0)}.
        std::array<T,4> Interpolate_Linearly(const T &at_x) const;

        //Returns linearly interpolated crossing-points.
        samples_1D<T> Crossings(T value) const;

        //Returns the locations linearly-interpolated peaks.
        samples_1D<T> Peaks(void) const;

        //Resamples the data into approximately equally-spaced samples using linear interpolation.
        samples_1D<T> Resample_Equal_Spacing(size_t N) const;

        //Binarize (threshold) the samples along x using Otsu's criteria.
        T Find_Otsu_Binarization_Threshold(void) const;

        //Multiply all sample f_i's by a given factor. Uncertainties are appropriately scaled too.
        samples_1D<T> Multiply_With(T factor) const; // i.e., "operator *".
        samples_1D<T> Sum_With(T factor) const; // i.e., "operator +".

        //Apply an absolute value functor to all f_i.
        samples_1D<T> Apply_Abs(void) const;

        //Shift all x_i's by a factor. No change to uncertainties.
        samples_1D<T> Sum_x_With(T dx) const;
        samples_1D<T> Multiply_x_With(T dx) const;
 
        //Distribution operators. Samples are retained, so linear interpolation is used.
        samples_1D<T> Sum_With(const samples_1D<T> &in) const; //i.e. "operator +".
        samples_1D<T> Subtract(const samples_1D<T> &in) const; //i.e. "operator -".

        //Purge non-finite samples.
        samples_1D<T> Purge_Nonfinite_Samples(void) const;

        //Generic driver for numerical integration. See source for more info.
        template <class Function> std::array<T,2> Integrate_Generic(const samples_1D<T> &in, Function F) const;

        //Computes the integral: \int_{-inf}^{+inf} f(x) dx and uncertainty estimate. See source.
        std::array<T,2> Integrate_Over_Kernel_unit() const;

        //Computes the integral: \int_{xmin}^{xmax} f(x) dx and uncertainty estimate. See source.
        std::array<T,2> Integrate_Over_Kernel_unit(T xmin, T xmax) const;

        //Computes integral over exp. kernel: \int_{xmin}^{xmax} f(x)*exp(A*(x+x0))dx and uncertainty estimate. See source.
        std::array<T,2> Integrate_Over_Kernel_exp(T xmin, T xmax, std::array<T,2> A, std::array<T,2> x0) const;

        //Group datum into (N) equal-dx bins, reducing variance. Ignores sigma_x_i. *Not* a histogram! See source.
        samples_1D<T> Aggregate_Equal_Sized_Bins_Weighted_Mean(long int N, bool explicitbins = false) const;

        //Group datum into equal-datum (N datum per bin) bins, reducing variance. *Not* a histogram! See source.
        samples_1D<T> Aggregate_Equal_Datum_Bins_Weighted_Mean(long int N) const; //Handles all uncertainties.

        //Generate histogram with (N) equal-dx bins. Bin height is sum of f_i in bin. Propagates sigma_f_i, ignores sigma_x_i.
        samples_1D<T> Histogram_Equal_Sized_Bins(long int N, bool explicitbins = false) const;

        //Spearman's Rank Correlation Coefficient. Ignores uncertainties. Returns: {rho, # of datum, z-value, t-value}.
        std::array<T,4> Spearmans_Rank_Correlation_Coefficient(bool *OK=nullptr) const;

        //Computes {Sxy,Sxx,Syy} which are used for linear regression and other procedures.
        std::array<T,3> Compute_Sxy_Sxx_Syy(bool *OK=nullptr) const;

        //Standard linear (y=m*x+b) regression. Ignores all provided uncertainties. Prefer weighted regression. See source.
        lin_reg_results<T> Linear_Least_Squares_Regression(bool *OK=nullptr, bool SkipExtras = false) const;

        //Calculates the discrete derivative using forward finite differences. (The right-side endpoint uses backward 
        // finite differences to handle the boundary.) Data should be reasonably smooth -- no interpolation is used.
        samples_1D<T> Derivative_Forward_Finite_Differences(void) const;

        //Calculates the discrete derivative using backward finite differences. (The right-side endpoint uses forward 
        // finite differences to handle the boundary.) Data should be reasonably smooth -- no interpolation is used.
        samples_1D<T> Derivative_Backward_Finite_Differences(void) const;

        //Calculates the discrete derivative using centered finite differences. (The endpoints use forward and backward 
        // finite differences to handle boundaries.) Data should be reasonably smooth -- no interpolation is used.
        samples_1D<T> Derivative_Centered_Finite_Differences(void) const;

    
        //Checks if the key is present without inspecting the value.
        bool MetadataKeyPresent(std::string key) const;

        //Attempts to cast the value if present. Optional is disengaged if key is missing or cast fails.
        //template <class U> std::experimental::optional<U> GetMetadataValueAs(std::string key) const;


        //Various IO routines.
        bool Write_To_File(const std::string &filename) const; //Writes data to file as 4 columns. Use it to plot/fit.
        std::string Write_To_String(void) const; //Writes data to file as 4 columns. Use it to plot/fit.

        bool Read_From_File(const std::string &filename); //Reads data from a file as 4 columns. True iff all went OK.

        void Plot(const std::string &Title = "") const; //Spits out a default plot of the data. 
        void Plot_as_PDF(const std::string &Title = "",const std::string &Filename = "/tmp/plot.pdf") const; //May overwrite existing files!

        //----------------------------------------------------- Friends ----------------------------------------------------
        //Overloaded stream operators for reading, writing, and serializing the samples.
        template<class Y> friend std::ostream & operator << (std::ostream &, const samples_1D<Y> &);
        template<class Y> friend std::istream & operator >> (std::istream &, samples_1D<Y> &);
};



}


#endif
