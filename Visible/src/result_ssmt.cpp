//
//  result_ssmt.cpp
//  Visible
//
//  Created by Arman Garakani on 8/20/18.
//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomma"
#pragma GCC diagnostic ignored "-Wunused-private-field"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <sstream>
#include <map>
#include <set>
#include "timed_types.h"
#include "core/signaler.h"
#include "sm_producer.h"
#include "cinder_cv/cinder_xchg.hpp"
#include "cinder_cv/cinder_opencv.h"
#include "vision/histo.h"
#include "vision/opencv_utils.hpp"
#include "ssmt.hpp"
#include "logger/logger.hpp"
#include "result_serialization.h"
#include "core/boost_stats.hpp"
#include "algo_runners.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace svl;

using cc_t = contractionLocator::contractionContainer_t;

////////// ssmt_result Implementataion //////////////////
const ssmt_result::ref_t ssmt_result::create (std::shared_ptr<ssmt_processor>& parent, const moving_region& child,
                                              const result_index_channel_t& in){
    ssmt_result::ref_t this_child (new ssmt_result(child, in ));
    this_child->m_weak_parent = parent;
	this_child->m_magnification_x = parent->parameters().magnification();
    return this_child;
}

ssmt_result::ssmt_result(const moving_region& mr,const result_index_channel_t& in):moving_region(mr),  m_input(in) {
    // Create a contraction object
	m_caRef = contractionLocator::create (in, mr.id());
	m_leveler.initialize(in, mr.id());
	
    // Suport ssmt_processor::Contraction Analyzed
	try{
		std::function<void (contractionLocator::contractionContainer_t&,const result_index_channel_t& in)>ca_analyzed_cb =
			boost::bind (&ssmt_result::contraction_ready, this, boost::placeholders::_1, boost::placeholders::_2);
		boost::signals2::connection ca_connection = m_caRef->registerCallback(ca_analyzed_cb);
	}
	catch (const std::exception & ex)
	{
		std::cout << " Throw in Binding Signals " << ex.what() << std::endl;
	}
	m_images_loaded.store(false, std::memory_order_release);
}

// When contraction is ready, signal a copy
void ssmt_result::contraction_ready (contractionLocator::contractionContainer_t& contractions,const result_index_channel_t& in)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Call the contraction available cb if any
    auto shared_parent = m_weak_parent.lock();
    if (shared_parent && shared_parent->signal_contraction_ready && shared_parent->signal_contraction_ready->num_slots() > 0)
    {
        contractionLocator::contractionContainer_t copied = m_caRef->contractions();
        shared_parent->signal_contraction_ready->operator()(copied, in);
    }

    vlogger::instance().console()->info(" Contractions Analyzed: ");
}

const result_index_channel_t& ssmt_result::input() const { return m_input; }

const vector<float>& ssmt_result::entropies () const { return m_entropies; }


size_t ssmt_result::Id() const { return id(); }
const std::shared_ptr<contractionLocator> & ssmt_result::locator () const { return m_caRef; }

const ssmt_processor::channel_vec_t& ssmt_result::content () const { return m_all_by_channel; }


bool  ssmt_result::generateRegionImages () const{
	if (m_images_loaded == false){
		bool done = get_channels(m_input.section());
		m_images_loaded.store(done, std::memory_order_release);
	}
	return m_images_loaded;
}

bool ssmt_result::run_selfsimilarity (){
	if (m_pci_done == false){
		auto pci_done = true; //run_selfsimilarity_on_region(m_all_by_channel[m_input.section()]);
		m_pci_done.store(pci_done, std::memory_order_release);
	}
	return m_pci_done;
		
}

bool ssmt_result::process (){

	while(!m_images_loaded || ! m_pci_done)
		{std::this_thread::yield(); }
	
	if (m_images_loaded == false || m_pci_done == false) return false;
//	m_leveled = m_leveler.leveledF();
	auto parent = m_weak_parent.lock();
	if (parent.get() == 0) return false;
	
	m_caRef->load(parent->entropies_F(), parent->ssMatrix());
    m_caRef->locate_contractions();

	auto ss_done = run_scale_space(m_all_by_channel[m_input.section()]);
	if(ss_done){
		// @todo: this should get a contraction to use
		m_scale_space.process_motion_peaks(0, motion_surface().boundingRect());
	}
	if(m_pci_done && ss_done){
		m_caRef->profile_contractions(m_scale_space.lengths());
		return true;
	}
	return false;
}

// Crops the moving body accross the sequence
bool ssmt_result::get_channels (int channel) const {
    
    auto parent = m_weak_parent.lock();
    if (parent.get() == 0) return false;
    m_channel_count = parent->channel_count();
    assert(parent->channel_count() > 0);
    m_all_by_channel.resize (m_channel_count);
    
    assert(channel>=0 && channel < m_channel_count);
    const vector<roiWindow<P8U> >& rws = parent->content()[channel];
    
    
    auto affineCrop = [] (const vector<roiWindow<P8U> >::const_iterator& rw_src, const cv::RotatedRect& rect){
        
        
        auto matAffineCrop = [] (Mat input, const RotatedRect& box){
            double angle = box.angle;
            auto size = box.size;
            
            auto transform = getRotationMatrix2D(box.center, angle, 1.0);
            Mat rotated;
            warpAffine(input, rotated, transform, input.size(), INTER_CUBIC);
            
            //Crop the result
            Mat cropped;
            getRectSubPix(rotated, size, box.center, cropped);
            copyMakeBorder(cropped,cropped,0,0,0,0,BORDER_CONSTANT,Scalar(0));
            roiWindow<P8U> r8(cropped.cols, cropped.rows);
            cpCvMatToRoiWindow8U (cropped, r8);
            return r8;
        };
        
        cvMatRefroiP8U(*rw_src, src, CV_8UC1);
        return matAffineCrop(src, rect);
        
    };
    
    
    vector<roiWindow<P8U> >::const_iterator vitr = rws.begin();
    uint32_t count = 0;
    uint32_t total = static_cast<uint32_t>(rws.size());
    do
    {
        auto rr = rotated_roi();
     //   rr.size.width += 40;
    //    rr.size.height += 20;
        auto affine = affineCrop(vitr, rr);
        m_all_by_channel[channel].emplace_back(affine);
        count++;
    }
    while (++vitr != rws.end());
    bool check = m_all_by_channel[channel].size() == total;
	
    check = check && count == total;
	return check;
}

bool ssmt_result::run_scale_space (const std::vector<roiWindow<P8U>>& images){
	
	return m_scale_space.generate(images,  2, 15, 2, m_magnification_x);
}

// Run to get Entropies and Median Level Set
// PCI track is being used for initial emtropy and median leveled
bool ssmt_result::run_selfsimilarity_on_region (const std::vector<roiWindow<P8U>>& images)
{
  
    auto sp =  std::shared_ptr<sm_producer> ( new sm_producer () );
	m_smat.resize(0);
	m_entropies.resize(0);
	
    vlogger::instance().console()->info(tostr(images.size()));
    sp->load_images (images);
    std::packaged_task<bool()> task([sp](){ return sp->operator()(0);}); // wrap the function
    std::future<bool>  future_ss = task.get_future();  // get a future
    std::thread(std::move(task)).join(); // Finish on a thread
    
    if (future_ss.get())
    {
        const deque<double> entropies = sp->shannonProjection ();
        const std::deque<deque<double>> sm = sp->similarityMatrix();
        assert(images.size() == entropies.size() && sm.size() == images.size());
        for (auto row : sm) assert(row.size() == images.size());
	
		// todo: remove all this nonsense copying.
		m_entropies.resize(entropies.size());
		std::transform(entropies.begin(), entropies.end(), m_entropies.begin(), [] (const double d){ return float(d); });
		std::vector<double> entropies_D(entropies.size());
		std::transform(entropies.begin(), entropies.end(), entropies_D.begin(), [] (const double d){ return d; });
	
	
        for (auto row : sm){
            vector<double> rowv;
            rowv.insert(rowv.end(), row.begin(), row.end());
            m_smat.push_back(rowv);
        }
	
		bool check = m_smat.size() == m_entropies.size();
		if (! check ) return check;
	
		
		m_leveler.load(entropies_D, m_smat);
		m_leveler.set_median_levelset_pct((m_leveler.parameters().range().first+m_leveler.parameters().range().second)/2.0f);
		
		
	
	    auto parent = m_weak_parent.lock();
	
		if (parent.get() != 0 && parent->signal_root_pci_ready){
			parent->signal_root_pci_ready->operator()(m_entropies, m_input);
		}
    
        return check;
    }
    return false;
}

