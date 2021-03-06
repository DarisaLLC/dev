#ifndef _CONTRACTION_H
#define _CONTRACTION_H

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <memory>
#include <typeinfo>
#include <string>
#include <tuple>
#include <chrono>
#include <numeric>
#include <iterator>
#include "core/pair.hpp"
#include "cardiomyocyte_model.hpp"
#include "core/stats.hpp"
#include "core/stl_utils.hpp"
#include "core/signaler.h"
#include "core/lineseg.hpp"
#include "eigen_utils.hpp"
#include "input_selector.hpp"
#include "timed_types.h"
#include "median_levelset.hpp"


using namespace std;
using namespace boost;
using namespace svl;

class contractionProfile;
typedef std::shared_ptr<contractionProfile> profileRef;
class contractionLocator;
typedef std::weak_ptr<contractionLocator> locatorWeakRef_t;


/* contractionMesh contains all measured data for a contraction
 * duration of a contraction is relaxaton_end - contraction_start
 * peak contraction is at contraction peak
 */

struct contractionMesh : public std::pair<size_t,size_t>
{

    //@note: pair representing frame number and frame time
    typedef std::pair<size_t,double> index_val_t;
    // container
    typedef std::vector<float> sigContainer_t;

    index_val_t contraction_start;
    index_val_t contraction_max_acceleration;
    index_val_t contraction_peak;
    index_val_t relaxation_max_acceleration;
    index_val_t relaxation_end;

    float contraction_peak_interpolated;
    
    double relaxation_visual_rank;
	
    double contraction_length;
	double relaxed_length;
	
    contraction_id_t m_bid;
    cell_id_t m_uid;
 
    sigContainer_t             elongation;
    sigContainer_t             normalized_length;
	sigContainer_t             force;
	sigContainer_t             length;
    
    cm::cardio_model               cardioModel;

    static size_t timeCount(const contractionMesh& a){
        return a.relaxation_end.first - a.contraction_start.first;
    }
    static size_t first(const contractionMesh& a){
        return a.contraction_start.first;
    }
    static size_t last (const contractionMesh& a){
        return a.relaxation_end.first;
    }
    
    
    static bool selfCheck (const contractionMesh& a){
        bool ok = a.relaxation_end.first > a.contraction_start.first;
        ok &=  a.contraction_peak.first > a.contraction_start.first;
        ok &=  a.contraction_peak.first < a.relaxation_end.first;
        auto fc = ok ? (a.relaxation_end.first - a.contraction_start.first) : 0;
        ok &= a.force.size() == size_t(fc);
        ok &= a.elongation.size() == size_t(fc);
        ok &= a.normalized_length.size() == size_t(fc);
        return ok;
        
    }
    
    static bool equal (const contractionMesh& left, const contractionMesh& right)
    {
        bool ok = left.contraction_start == right.contraction_start;
        if (! ok) return ok;
        ok &=  left.contraction_max_acceleration == right.contraction_max_acceleration;
        if (! ok) return ok;
        ok &=  left.contraction_peak == right.contraction_peak;
        if (! ok) return ok;
        ok &=  left.relaxation_max_acceleration == right.relaxation_max_acceleration;
        if (! ok) return ok;
        ok &=  left.relaxation_end == right.relaxation_end;
        return ok;
    }
    
    static void clear (contractionMesh& ct)
    {
        std::memset(&ct, 0, sizeof(contractionMesh));
    }
    
    static void fill (const index_val_t& peak, const std::pair<size_t,size_t>& start_end, contractionMesh& fillthis)
    {
        fillthis.contraction_peak = peak;
        fillthis.contraction_start.first = peak.first + start_end.first;
        fillthis.relaxation_end.first = peak.first + start_end.second;
    }

};

class ca_signaler : public base_signaler
{
    virtual std::string
    getName () const { return "caSignaler"; }
};



class contractionLocator : public ca_signaler, std::enable_shared_from_this<contractionLocator>
{
public:
    
    class params{
    public:
        params ():m_minimum_contraction_time (0.32), m_frame_duration(0.020),
		m_magnification_x(10.0f), m_min_peak_to_relaxation_end(5.0), m_beats(1) {
            update ();
        }
        float minimum_contraction_time () const { return m_minimum_contraction_time; }
        void minimum_contraction_time (const float newval) const {
            m_minimum_contraction_time = newval;
            update ();
        }
        float frame_duration () const { return m_frame_duration; }
        void frame_duration (float newval) const {
            m_frame_duration = newval;
            update ();
        }
        
		float minimum_peak_relaxation_end() const { return m_min_peak_to_relaxation_end; }
		void  minimum_peak_relaxation_end(float newval) const { m_min_peak_to_relaxation_end = newval; }
		
        uint32_t minimum_contraction_frames () const { return m_pad_frames; }
		
		void magnification (const float& mmag) const { m_magnification_x = mmag; }
		float magnification () const { return m_magnification_x; }
		
		void beats (const int& num) { m_beats = num; }
		int beats () { return m_beats; }
		
        
    private:
        void update () const {
            m_pad_frames = m_minimum_contraction_time / m_frame_duration;
        }
		mutable float m_magnification_x;
        mutable float m_minimum_contraction_time;
        mutable float m_frame_duration;
        mutable uint32_t m_pad_frames;
		mutable float m_min_peak_to_relaxation_end;
		mutable int m_beats;
    };
    
    using contraction_t = contractionMesh;
    using index_val_t = contractionMesh::index_val_t;
    using sigContainer_t = contractionMesh::sigContainer_t;
  
    typedef std::shared_ptr<contractionLocator> Ref;
    typedef std::vector<contraction_t> contractionContainer_t;

    // Signals we provide
    // signal_contraction_available

    typedef void (sig_cb_contraction_ready) (contractionContainer_t&, const result_index_channel_t& );
    typedef void (sig_cb_cell_length_ready) (sigContainer_t&);
    typedef void (sig_cb_force_ready) (sigContainer_t&);
    
    // Factory create method
    static Ref create(const result_index_channel_t&,  const uint32_t& body_id, const contractionLocator::params& params = contractionLocator::params());
    Ref getShared();
    locatorWeakRef_t getWeakRef();
    
    // Load raw entropies and the self-similarity matrix
    // If no self-similarity matrix is given, entropies are assumed to be filtered and used directly
    // // input selector -1 entire index mobj index
    void load (const vector<float>& entropies, const vector<vector<double>>& mmatrix = vector<vector<double>>());
	
	const contractionLocator::params& parameters () const { return m_params; }

 
    
    bool locate_contractions () ;
    bool profile_contractions (const std::vector<float>& lengths = std::vector<float>());

    
    size_t size () const { return m_entsize; }
    const uint32_t id () const { return m_id; }
    const result_index_channel_t& input () const { return m_in; }
    
    // Original
    const vector<float>& entropies () { return m_entropies; }
    
    const std::vector<index_val_t>& contraction_peaks () const { return m_peaks; }

    const std::vector<float>& contraction_interpolated_peaks () const { return m_peaks_interpolated; }
    
    const std::vector<int>& contraction_peak_frame_indicies () const { return m_peaks_idx; }
    
    const contractionContainer_t & contractions () const { return m_contractions; }
    
  
private:
    contractionLocator(const result_index_channel_t&,  const uint32_t& body_id, const contractionLocator::params& params = contractionLocator::params ());
    contractionLocator::params m_params;
	
	int lookup_peak_location(const int peak_index) const;
	
	bool get_contraction_at_point (int src_peak_index, const std::vector<double>& peaks_locations, contraction_t& ) const;


    mutable double m_median_value;
    mutable vector<vector<double>>        m_SMatrix;   // Used in eExhaustive and
    vector<float>               m_entropies;

    mutable vector<float>              m_signal;
    std::vector<int> m_peaks_idx;
	std::vector<float> m_peaks_fidx;

    size_t m_entsize;
    mutable bool mNoSMatrix;
    mutable std::vector<index_val_t> m_peaks;
	mutable std::vector<float> m_peaks_interpolated;
	mutable std::vector<double> m_peaksLoc;
	mutable std::vector<double> m_peaksVal;
	
    mutable contractionContainer_t m_contractions;
    mutable int m_input; // input source
    mutable int m_id;
    mutable std::vector<double> m_ac;
    mutable std::vector<double> m_bac;
    result_index_channel_t m_in;
    
protected:
    boost::signals2::signal<contractionLocator::sig_cb_cell_length_ready>* cell_length_ready;
    boost::signals2::signal<contractionLocator::sig_cb_force_ready>* total_reactive_force_ready;
    boost::signals2::signal<contractionLocator::sig_cb_contraction_ready>* contraction_ready;

    
};



class contractionProfile
{
public:
    using contraction_t = contractionMesh;
    using index_val_t = contractionMesh::index_val_t;
    using sigContainer_t = contractionMesh::sigContainer_t;
    
    contractionProfile (contraction_t&, const std::vector<float>& signal, const std::vector<float>& lengths = std::vector<float>());

    static profileRef create(contraction_t& ct, const std::vector<float>& signal, const std::vector<float>& lengths = std::vector<float>()){
        return profileRef(new contractionProfile(ct, signal, lengths));
    }
    // Compute Length Interpolation for measured contraction
    bool compute_interpolated_geometries_and_force();
    
    const contraction_t& contraction () const { return m_ctr; }
    const double& relaxed_length () const { return m_relaxed_length; }
    const sigContainer_t& profiled () const { return m_fder; }
    const sigContainer_t& force () const { return m_force; }
    const sigContainer_t& interpolatedLength () const { return m_interpolated_length; }
	const sigContainer_t& elongation () const { return m_elongation; }
	const sigContainer_t& lengths () const { return m_lengths; }
    
    
private:
    typedef vector<double>::iterator dItr_t;
    lsFit<double> m_line_fit;
    std::pair<uRadian,double> m_ls_result;
    mutable contraction_t m_ctr;
    double m_relaxed_length;
	mutable std::vector<float> m_fder;
	mutable std::vector<float> m_lengths;
    cell_id_t m_id;
    mutable sigContainer_t m_interpolated_length;
    mutable sigContainer_t m_elongation;
    mutable sigContainer_t m_force;

};



#endif





