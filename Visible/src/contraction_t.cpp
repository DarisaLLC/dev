//
//  contraction.cpp
//  Visible
//
//  Created by Arman Garakani on 8/19/18.
//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomma"
#pragma GCC diagnostic ignored "-Wunused-private-field"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "cinder_cv/cinder_opencv.h"

#include "contraction.hpp"
#include "sg_filter.h"
#include "core/core.hpp"
#include "core/stl_utils.hpp"
#include "core/fit.hpp"
#include "logger/logger.hpp"
#include "cardiomyocyte_model_detail.hpp"
#include "core/boost_units.hpp"
#include "core/simple_timing.hpp"
#include "core/moreMath.h"
#include "core/stats.hpp"
#include "core/pf.h"
#include "vision/correlation1d.hpp"
#include "core/boost_stats.hpp"
#include "logger/logger.hpp"
#include "cpp-perf.hpp"
#include "boost/filesystem.hpp"
#include "thread.h"

using namespace perf;
using namespace boost;

namespace bfs=boost::filesystem;

namespace anonymous
{
    recursive_mutex contraction_mutex;
}




std::shared_ptr<contractionLocator> contractionLocator::create(const input_section_selector_t& in, const uint32_t& body_id, const contractionLocator::params& params){
    recursive_lock_guard lock(anonymous::contraction_mutex);
    return std::shared_ptr<contractionLocator>(new contractionLocator(in, body_id, params));
}

std::shared_ptr<contractionLocator> contractionLocator::getShared () {
    return shared_from_this();
}

locatorWeakRef_t contractionLocator::getWeakRef(){
    locatorWeakRef_t weak = getShared();
    return weak;
}

contractionLocator::contractionLocator(const input_section_selector_t& in,  const uint32_t& body_id, const contractionLocator::params& params) :
m_cached(false),  m_in(in), mValidInput (false), m_params(params)
{
    m_median_levelset_frac = m_params.median_levelset_fraction();
    m_peaks.resize(0);
    m_id = body_id;
    m_in = in;
    // Signals we provide
    contraction_ready = createSignal<contractionLocator::sig_cb_contraction_ready> ();
    med_pci_ready = createSignal<contractionLocator::sig_cb_med_pci_ready> ();
    total_reactive_force_ready = createSignal<contractionLocator::sig_cb_force_ready>();
    cell_length_ready = createSignal<contractionLocator::sig_cb_cell_length_ready>();
}

void contractionLocator::load(const vector<float>& entropies, const vector<vector<double>>& mmatrix)
{
    m_entropies = entropies;
    m_SMatrix = mmatrix;
    mNoSMatrix = m_SMatrix.empty();
    
    m_entsize = m_entropies.size();
    if (isPreProcessed()) m_signal = m_entropies;
    else{
        m_ranks.resize (m_entsize);
        m_signal.resize (m_entsize);
    }
    m_peaks.resize(0);
    mValidInput = verify_input ();
    auto msg = "Contraction Locator Initialized For " + to_string(m_id);
    vlogger::instance().console()->info(msg);
}

void contractionLocator::compute_median_levelsets () const
{
	if (m_cached) return;
	m_median_value = contractionLocator::Median_levelsets (m_entropies, m_ranks);
	m_cached = true;
}

/* Compute rank for all entropies
 * Steps:
 * 1. Copy and Calculate Median Entropy
 * 2. Subtract Median from the Copy
 * 3. Sort according to distance to Median
 * 4. Fill up rank vector
 */
double contractionLocator::Median_levelsets (const vector<float>& entropies,  std::vector<int>& ranks )
{
	perf::timer timeit;
	timeit.start();
	
    vector<float> entcpy;
    
    for (const auto val : entropies)
        entcpy.push_back(val);
    auto median_value = svl::Median(entcpy);
    // Subtract median from all
    for (float& val : entcpy)
        val = std::abs (val - median_value );
    
    // Sort according to distance to the median. small to high
    ranks.resize(entropies.size());
    std::iota(ranks.begin(), ranks.end(), 0);
    auto comparator = [&entcpy](float a, float b){ return entcpy[a] < entcpy[b]; };
    std::sort(ranks.begin(), ranks.end(), comparator);
    return median_value;
	timeit.stop();
	auto timestr = toString(std::chrono::duration_cast<milliseconds>(timeit.duration()).count());
	vlogger::instance().console()->info(" Median Level Set (ms): " + timestr);
}

size_t contractionLocator::recompute_signal () const
{

    size_t count = std::floor (m_entropies.size () * m_median_levelset_frac);
    assert(count < m_ranks.size());
    m_signal.resize(m_entropies.size (), 0.0);
    for (auto ii = 0; ii < m_signal.size(); ii++)
    {
        double val = 0;
        for (int index = 0; index < count; index++)
        {
            auto jj = m_ranks[index]; // get the actual index from rank
            val += m_SMatrix[jj][ii]; // fetch the cross match value for
        }
        val = val / count;
        m_signal[ii] = val;
    }
    
    return count;
}
void contractionLocator::update() const{
	if (m_entropies.empty()) return;
	
    // Cache Rank Calculations
    compute_median_levelsets ();
    // If the fraction of entropies values expected is zero, then just find the minimum and call it contraction
    size_t count = recompute_signal();
    if (count == 0) m_signal = m_entropies;
	
	if (med_pci_ready && med_pci_ready->num_slots() > 0)
		med_pci_ready->operator()(m_signal);
 
}

bool contractionLocator::get_contraction_at_point (int src_peak_index, const std::vector<int>& peak_indices, contraction_t& m_contraction) const{
    perf::timer timeit;
    timeit.start();
      
    if (! mValidInput) return false;
    contraction_t::clear(m_contraction);
    
    auto maxima = [](std::vector<float>::const_iterator a,
                     std::vector<float>::const_iterator b){
        std::vector<double> coeffs;
        polyfit(a, b, 1.0, coeffs, 2);
        assert(coeffs.size() == 3);
        return (-coeffs[1] / (2 * coeffs[2]));
    };
    const std::vector<float>& m_fder = m_signal;
    
    // Find distance between this peak and its previous and next peak or the begining or the end of the signal
    
    /* Contraction Start
     * 1. Find the location of maxima of a quadratic fit to the data before
     *    Maxima is at -b / 2c (y = a + bx + cx^2 => dy/dx = b + 2cx at Maxima dy/dx = 0 -> x = -b/2c
     * 1. Find the location of the minima of the first derivative using first differences
     *    dy = y(x-1) - y(x). x at minimum dy
     *
     */
    bool edge_case = peak_indices[src_peak_index] < m_params.minimum_contraction_frames() ||
        (m_fder.size() - peak_indices[src_peak_index]) < m_params.minimum_contraction_frames() ;
    if (edge_case) return false;
    
    m_contraction.contraction_peak.first = peak_indices[src_peak_index];
    std::vector<float>::const_iterator cpt = m_fder.begin() + peak_indices[src_peak_index];
    std::vector<float> results;
    // Left Boundary: If there is a peak behind us, use that other wise signal start
    auto left_boundary = src_peak_index > 0 ? peak_indices[src_peak_index - 1] : 0;
    results.reserve(m_contraction.contraction_peak.first - left_boundary);
    std::adjacent_difference (m_fder.begin()+left_boundary, cpt, std::back_inserter(results));
    auto min_elem = std::min_element(results.begin(),results.end());
    auto loc = std::distance(results.begin(), min_elem);
    
    auto loc_contraction_quadratic = maxima(m_fder.begin()+left_boundary, cpt);
    auto value = (! std::signbit(loc_contraction_quadratic))
        ? std::max(size_t(loc_contraction_quadratic), size_t(loc)) - std::min(size_t(loc_contraction_quadratic), size_t(loc)) / 2
        : loc;
    m_contraction.contraction_start.first = value;
    
    auto right_boundary = (src_peak_index == (peak_indices.size() - 1))
        ? m_fder.size() - 1
        : peak_indices[src_peak_index + 1];
    /* relaxation End
     * 1. Find the location of maxima of a quadratic fit to the data before
     *    Maxima is at -b / 2c (y = a + bx + cx^2 => dy/dx = b + 2cx at Maxima dy/dx = 0 -> x = -b/2c
     */
    auto loc_relaxation_quadratic = maxima(cpt, m_fder.begin()+right_boundary);
    m_contraction.relaxation_end.first =  m_contraction.contraction_peak.first + loc_relaxation_quadratic;
	
	// Make sure that relaxation end is beyond conraction peak 
    if (m_contraction.relaxation_end.first < (m_contraction.contraction_peak.first + m_params.minimum_peak_relaxation_end()))
		return false;
	
    m_contraction.contraction_peak_interpolated = m_contraction.contraction_peak.first;
    m_contraction.contraction_peak_interpolated += parabolicFit(1.0 - m_fder[m_contraction.contraction_peak.first-1],
                                                               1.0 - m_fder[m_contraction.contraction_peak.first],
                                                               1.0 - m_fder[m_contraction.contraction_peak.first+1]);

    m_peaks_interpolated.push_back(m_contraction.contraction_peak_interpolated);

    /*
     * Use Median visual rank value for relaxation visual rank
     */
    m_contraction.relaxation_visual_rank = svl::Median(m_fder);
    timeit.stop();
    auto timestr = toString(std::chrono::duration_cast<milliseconds>(timeit.duration()).count());
    vlogger::instance().console()->info(" get_contraction took (ms): " + timestr);
    vlogger::instance().console()->info(" Peak Index: " + toString(m_contraction.contraction_peak.first));
    vlogger::instance().console()->info(" Peak Interpolated: " + toString(m_contraction.contraction_peak_interpolated));
    
    return true;
}



bool contractionLocator::locate_contractions (){

    assert(verify_input());
    if(! isPreProcessed())
        update ();
    
    m_ac.resize(0);
    f1dAutoCorr(m_signal.begin(), m_signal.end(), m_ac);
    
    m_bac.resize(m_ac.size());
    cv::GaussianBlur(m_ac,m_bac,cv::Size(0,0), 4.0);
    {
        stringstream ss;

        svl::boostStatistics bstats;
        for (const auto& val : m_bac) bstats.update(val);
        ss << std::endl;
        ss << "(time in ms)" << std::endl
        << "Count:  " << bstats.get_n() << std::endl
        << "Min:    " << bstats.get_min() << std::endl
        << "Max:    " << bstats.get_max() << std::endl
        << "Med:    " << bstats.get_median() << std::endl
        << "x75:    " << bstats.get_quantile(0.75) << std::endl
        << "x85:    " << bstats.get_quantile(0.85) << std::endl
        << "x95:    " << bstats.get_quantile(0.95) << std::endl
        << "x99:    " << bstats.get_quantile(0.99) << std::endl
        << "x99.9:  " << bstats.get_quantile(0.999) << std::endl
        << "x99.99: " << bstats.get_quantile(0.9999) << std::endl;
        vlogger::instance().console()->info(ss.str());
    }
    
    clear_outputs ();
    m_peaks_idx.resize(0);
    
    // @to_remove Flip for peak finding
    std::vector<double> valleys(m_signal.size());
    std::transform(m_signal.begin(), m_signal.end(), valleys.begin(), [](double f)->double { return 1.0 - f; });
    
    svl::findPeaks(valleys, m_peaks_idx);
    // Find periods
    std::vector<int> periods;
    std::adjacent_difference (m_peaks_idx.begin(), m_peaks_idx.end(), std::back_inserter(periods));
    auto median_period = svl::Median(periods);
    auto half_period = median_period / 2;
    std::vector<int> selected;
    for (int ii = 0; ii < m_peaks_idx.size(); ii++){
        if (periods[ii] > half_period)
            selected.push_back(m_peaks_idx[ii]);
    }
    {
        stringstream ss;
        for (auto idx : m_peaks_idx) ss << idx << "," << std::endl;
        for (auto idx : periods) ss << idx << "," << std::endl;
        for (auto idx : selected) ss << idx << "," << std::endl;
        vlogger::instance().console()->info(ss.str());
    }
    m_peaks_idx = selected;
    
    // @todo check if this and peak indexes are redundant in use
    for (auto pp = 0; pp < m_peaks_idx.size(); pp++){
        m_peaks.emplace_back(m_peaks_idx[pp], m_signal[m_peaks_idx[pp]]);
    }
    return true;
}
        // Note: Lengths are in units of microns.
		// lengthFromMotion is created with Magnification X
		// And converts accordingly
bool contractionLocator::profile_contractions (const std::vector<float>& lengths){
    perf::timer timeit;
    timeit.start();
    
    for (auto pp = 0; pp < m_peaks_idx.size(); pp++){
        
        contraction_t ct;
        ct.m_uid = id();
        ct.m_bid = pp;
        
        // Contraction gets a unique id by contraction profiler
        if(get_contraction_at_point(pp, m_peaks_idx, ct)){
            auto profile = contractionProfile::create(ct, m_signal, lengths);
			if ( ! profile->compute_interpolated_geometries_and_force()) continue;
			
            m_contractions.emplace_back(profile->contraction());
//          @todo progress & report 
        }
    }
    
    {
        stringstream ss;
        for (auto pinterp : m_peaks_interpolated) ss << pinterp << ",";
        vlogger::instance().console()->info(ss.str());
    }
    
    if (contraction_ready && contraction_ready->num_slots() > 0)
        contraction_ready->operator()(m_contractions, m_in);

    timeit.stop();
    auto timestr = toString(std::chrono::duration_cast<milliseconds>(timeit.duration()).count());
    vlogger::instance().console()->info(" Update (ms): " + timestr);
    vlogger::instance().console()->info(" Number of Peaks: " + toString(m_peaks_idx.size()));
    
    return true;
}



void contractionLocator::clear_outputs () const
{
    m_peaks.resize(0);
    m_contractions.resize(0);
}
bool contractionLocator::verify_input () const
{
    bool ok = ! m_entropies.empty();
    if (!ok)return false;
    
    if ( ! m_SMatrix.empty () ){
        ok = m_entropies.size() == m_entsize;
        if (!ok) return ok;
        ok &= m_SMatrix.size() == m_entsize;
        if (!ok) return ok;
        for (int row = 0; row < m_entsize; row++)
        {
            //      std::cout << "[" << row << "]" << m_SMatrix[row].size () << std::endl;
            ok &= m_SMatrix[row].size () == m_entsize;
            if (!ok) return ok;
        }
    }
    return ok;
}





template boost::signals2::connection contractionLocator::registerCallback(const std::function<contractionLocator::sig_cb_med_pci_ready>&);
template boost::signals2::connection contractionLocator::registerCallback(const std::function<contractionLocator::sig_cb_contraction_ready>&);

contractionProfile::contractionProfile (contraction_t& ct, const std::vector<float>& signal,const std::vector<float>& lengths ):
m_ctr(ct), m_lengths(lengths)
{
	for (const auto dd : signal) m_fder.push_back((float)dd);
}




/*
 * Compute interpolated length and force for this contraction
 *
 */

bool contractionProfile::compute_interpolated_geometries_and_force(){
    perf::timer timeit;
    timeit.start();
	
	
    double relaxed_length = m_ctr.relaxation_visual_rank;
	if (!m_lengths.empty()){
		relaxed_length = Median(m_lengths);
	}
	if(m_fder.empty()) return false;

    /*
     * Compute everything within contraction interval
     */
    auto c_start = m_ctr.contraction_start.first;
    auto c_end = std::min(m_ctr.relaxation_end.first, m_fder.size()-1);

	bool check = c_start >= 0 && c_start < c_end && c_end < m_fder.size();
	if (! check) return check;
    
    // Compute force using cardio model
    m_elongation.clear();
    m_interpolated_length.clear();
    m_force.clear();
    cardio_model cmm;
    cmm.shear_control(1.0f);
    cmm.shear_velocity(200.0_cm_s);
    m_ctr.cardioModel = cmm;
	if (!m_lengths.empty()){
		auto minIter = std::min_element(m_lengths.begin(), m_lengths.end());
		if (minIter != m_lengths.end())
			m_ctr.contraction_length = *minIter;
	}
	
    for (auto ii = c_start; ii < c_end; ii++)
    {
        auto linMul = clampValue(m_fder[ii]/relaxed_length, 0.0, 1.0);
        auto elon = relaxed_length - linMul;
		if (! m_lengths.empty()){
			linMul = m_lengths[ii];
			elon = relaxed_length - linMul;
		}
	
        cmm.length(linMul * boost::units::cgs::micron);
        cmm.elongation(elon * boost::units::cgs::micron);
        cmm.width((linMul/3.0) * boost::units::cgs::micron);
        cmm.thickness((linMul/100.0)* boost::units::cgs::micron);
        cmm.run();
        m_force.push_back(cmm.result().total_reactive.value());
        m_interpolated_length.push_back(linMul);
        m_elongation.push_back(elon);
    }
    
    // Copy Results Out.
    m_ctr.force = m_force;
    m_ctr.elongation = m_elongation;
    m_ctr.normalized_length = m_interpolated_length;
	m_ctr.length = m_lengths;
	m_ctr.relaxed_length = relaxed_length;

    
    timeit.stop();
    auto timestr = toString(std::chrono::duration_cast<milliseconds>(timeit.duration()).count());
    vlogger::instance().console()->info("Force took (ms): " + timestr);
    vlogger::instance().console()->info("Cell Id " + toString(m_ctr.m_uid) + " Contraction Id " + toString(m_ctr.m_bid));
	return true;
    
}

#pragma GCC diagnostic pop
