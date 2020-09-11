//
//  result_ssmt.cpp
//  Visible
//
//  Created by Arman Garakani on 8/20/18.
//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomma"
#pragma GCC diagnostic ignored "-Wunused-private-field"

#include "opencv2/stitching.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <sstream>
#include <map>
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


using cc_t = contractionLocator::contractionContainer_t;

////////// ssmt_result Implementataion //////////////////
const ssmt_result::ref_t ssmt_result::create (std::shared_ptr<ssmt_processor>& parent, const moving_region& child,
                                              const input_channel_selector_t& in){
    ssmt_result::ref_t this_child (new ssmt_result(child, in ));
    this_child->m_weak_parent = parent;
    return this_child;
}

ssmt_result::ssmt_result(const moving_region& mr,const input_channel_selector_t& in):moving_region(mr),  m_input(in) {
    // Create a contraction object
    m_caRef = contractionLocator::create (in, mr.id());
    
    // Suport lif_processor::Contraction Analyzed
    std::function<void (contractionLocator::contractionContainer_t&,const input_channel_selector_t& in)>ca_analyzed_cb =
    boost::bind (&ssmt_result::contraction_ready, this, boost::placeholders::_1, boost::placeholders::_2);
    boost::signals2::connection ca_connection = m_caRef->registerCallback(ca_analyzed_cb);
}

// When contraction is ready, signal a copy
void ssmt_result::contraction_ready (contractionLocator::contractionContainer_t& contractions,const input_channel_selector_t& in)
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

const input_channel_selector_t& ssmt_result::input() const { return m_input; }

size_t ssmt_result::Id() const { return id(); }
const std::shared_ptr<contractionLocator> & ssmt_result::locator () const { return m_caRef; }

const ssmt_processor::channel_vec_t& ssmt_result::content () const { return m_all_by_channel; }


void ssmt_result::process (){
    bool done = get_channels(m_input.channel());
    if(done){
        auto pci_done = run_selfsimilarity_on_region(m_all_by_channel[m_input.channel()]);
        if(pci_done) m_caRef->locate_contractions();
    }
}

// Crops the moving body accross the sequence
bool ssmt_result::get_channels (int channel){
    
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
        const auto& rr = rotated_roi();
        auto affine = affineCrop(vitr, rr);
        m_all_by_channel[channel].emplace_back(affine);
        count++;
    }
    while (++vitr != rws.end());
    bool check = m_all_by_channel[channel].size() == total;
    return check && count == total;
}


// Run to get Entropies and Median Level Set
// PCI track is being used for initial emtropy and median leveled
bool ssmt_result::run_selfsimilarity_on_region (const std::vector<roiWindow<P8U>>& images)
{
  
    auto sp =  std::shared_ptr<sm_producer> ( new sm_producer () );
    
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
        
        m_entropies.insert(m_entropies.end(), entropies.begin(), entropies.end());
        for (auto row : sm){
            vector<double> rowv;
            rowv.insert(rowv.end(), row.begin(), row.end());
            m_smat.push_back(rowv);
        }
        
        m_caRef->load(m_entropies, m_smat);
        return true;
    }
    return false;
}
