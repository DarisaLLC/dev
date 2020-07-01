//
//  voxel_segmentation.cpp
//  Visible
//
//  Created by Arman Garakani on 5/26/19.
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
#include "segmentation_parameters.hpp"

using namespace stl_utils;
using namespace boost;
namespace bfs=boost::filesystem;


// Return 2D latice of pixels over time
void ssmt_processor::generateVoxels_on_channel (const int channel_index){
    generateVoxelsOfSampled(m_all_by_channel[channel_index]);
}


// Return 2D latice of voxel self-similarity

void ssmt_processor::generateVoxelsAndSelfSimilarities (const std::vector<roiWindow<P8U>>& images){
    
    generateVoxelsOfSampled(images);
    generateVoxelSelfSimilarities();

}



void ssmt_processor::generateVoxelsOfSampled (const std::vector<roiWindow<P8U>>& images){
    size_t t_d = images.size();
    auto half_offset = m_voxel_sample / 2;
    uint32_t width = images[0].width() - half_offset.first;
    uint32_t height = images[0].height() - half_offset.second;
    uint32_t expected_width = m_expected_segmented_size.first;
    uint32_t expected_height = m_expected_segmented_size.second;
    
    m_voxels.resize(0);
    std::string msg = " Generating Voxels @ (" + to_string(m_voxel_sample.first) + "," + to_string(m_voxel_sample.second) + ")";
    msg += "Expected Size: " + to_string(expected_width) + " x " + to_string(expected_height);
    vlogger::instance().console()->info("starting " + msg);
    
    int count = 0;
    int row, col;
    for (row = half_offset.second; row < height; row+=m_voxel_sample.second){
        for (col = half_offset.first; col < width; col+=m_voxel_sample.first){
            std::vector<uint8_t> voxel(t_d);
            for (auto tt = 0; tt < t_d; tt++){
                voxel[tt] = images[tt].getPixel(col, row);
            }
            count++;
            m_voxels.emplace_back(voxel);
        }
        expected_height -= 1;
    }
    expected_width -= (count / m_expected_segmented_size.second);
    
    bool ok = expected_width == expected_height && expected_width == 0;
    msg = " remainders: " + to_string(expected_width) + " x " + to_string(expected_height);
    vlogger::instance().console()->info(ok ? "finished ok " : "finished with error " + msg);
    
    
}
/*
   Create Entropy Space from voxel entropies
*/
void ssmt_processor::create_voxel_surface(std::vector<float>& ven){
    
    uiPair half_offset = m_voxel_sample / 2;
    uiPair size_diff = half_offset + half_offset;
    uint32_t width = m_expected_segmented_size.first;
    uint32_t height = m_expected_segmented_size.second;
    assert(width*height == ven.size());
    auto extremes = svl::norm_min_max(ven.begin(),ven.end());
    std::string msg = to_string(extremes.first) + "," + to_string(extremes.second);
    vlogger::instance().console()->info(msg);
    
    cv::Mat ftmp = cv::Mat(height, width, CV_32F);
    std::vector<float>::const_iterator start = ven.begin();
    std::vector<uint32_t> hist(256,0);

    
    // Fillup floating image and hist of 8bit values
    for (auto row =0; row < height; row++){
        float* f_row_ptr = ftmp.ptr<float>(row);
        auto end = start + width; // point to one after
        for (; start < end; start++, f_row_ptr++){
            if(std::isnan(*start)) continue; // nan is set to 1, the pre-set value
            float valf(*start);
            uint8_t val8 (valf*255);
            hist[val8]++;
            *f_row_ptr = valf;
        }
        start = end;
    }

    // Find the mode of the histogram
    // @note add validation
    auto hm = std::max_element(hist.begin(), hist.end());
    if(hm == hist.end()){
        vlogger::instance().console()->error("motion signal error");
        return;
    }

    double motion_peak(std::distance(hist.begin(), hm));

    // Median Blur before up sizeing.
    cv::medianBlur(ftmp, ftmp, 5);
    
    // Straight resize. Add pads afterwards
    cv::Size bot(ftmp.cols * m_voxel_sample.first,ftmp.rows * m_voxel_sample.second);
    cv::Mat f_temporal (bot, CV_32F);
    cv::resize(ftmp, f_temporal, bot, 0, 0, INTER_NEAREST);
    ftmp = getPadded(f_temporal, half_offset, 0.0);

    ftmp = ftmp * 255.0f;
    ftmp.convertTo(m_temporal_ss, CV_8U);
    
    if(m_var_peaks.size() == 0){
        vlogger::instance().console()->error("motion space flat" );
        return;
    }
    // Sum sim under the var peaks
    // @note: Temporal Info is collected at full res. Voxel segmentation is done at reduce res
    float peaks_sums = 0.0f;
    for (cv::Point& pt : m_var_peaks)
        peaks_sums += m_temporal_ss.at<uint8_t>(pt.y,pt.x);
    auto peaks_average = peaks_sums / int(m_var_peaks.size());
    
    msg = " Temporal Peak Average  @ (" + to_string(peaks_average) + ", Motion Peak " + to_string(motion_peak) + ")";
    vlogger::instance().console()->info("starting " + msg);
    
    m_measured_area = Rectf(0,0,bot.width, bot.height);
    assert(m_temporal_ss.cols == m_measured_area.getWidth() + size_diff.first);
    assert(m_temporal_ss.rows == m_measured_area.getHeight() + size_diff.second);
    
    
    cv::Mat bi_level;

    threshold(m_temporal_ss, bi_level, peaks_average, 255, THRESH_BINARY);
    
    vlogger::instance().console()->info("finished voxel surface");
    if(bfs::exists(mCurrentCachePath)){
        std::string imagename = "voxel_ss_.png";
        auto image_path = mCurrentCachePath / imagename;
        cv::imwrite(image_path.string(), m_temporal_ss);
    }
    
    // Call the voxel ready cb if any
    if (signal_segmented_view_ready && signal_segmented_view_ready->num_slots() > 0){
        signal_segmented_view_ready->operator()(m_temporal_ss, bi_level);
    }
    
    
}

void ssmt_processor::generateVoxelSelfSimilarities (){
    
    bool cache_ok = false;
    std::shared_ptr<internalContainer> ssref;
    
    if(bfs::exists(mCurrentCachePath)){
        auto cache_path = mCurrentCachePath / m_params.internal_container_cache_name ();
        if(bfs::exists(cache_path)){
            ssref = internalContainer::create(cache_path);
        }
        cache_ok = ssref && ssref->size_check(m_expected_segmented_size.first*m_expected_segmented_size.second);
    }
    
    if(cache_ok){
        vlogger::instance().console()->info(" IC container cache : Hit ");
        m_voxel_entropies.insert(m_voxel_entropies.end(), ssref->data().begin(),
                                 ssref->data().end());
        // Call the voxel ready cb if any
        if (signal_ss_voxel_ready && signal_ss_voxel_ready->num_slots() > 0)
            signal_ss_voxel_ready->operator()(m_voxel_entropies);
        
    }else{
        
        auto cache_path = mCurrentCachePath / m_params.internal_container_cache_name ();
        
        vlogger::instance().console()->info("starting generating voxel self-similarity");
        auto sp =  similarity_producer();
        sp->load_images (m_voxels);
        std::packaged_task<bool()> task([sp](){ return sp->operator()(0);}); // wrap the function
        std::future<bool>  future_ss = task.get_future();  // get a future
        std::thread(std::move(task)).join(); // Finish on a thread
        vlogger::instance().console()->info("dispatched voxel self-similarity");
        if (future_ss.get())
        {
            vlogger::instance().console()->info("copying results of voxel self-similarity");
            const deque<double>& entropies = sp->shannonProjection ();
            m_voxel_entropies.insert(m_voxel_entropies.end(), entropies.begin(), entropies.end());
            
            // Call the voxel ready cb if any
            if (signal_ss_voxel_ready && signal_ss_voxel_ready->num_slots() > 0)
                signal_ss_voxel_ready->operator()(m_voxel_entropies);
            
            bool ok = internalContainer::store(cache_path, m_voxel_entropies);
            if(ok)
                vlogger::instance().console()->info(" SS result container cache : filled ");
            else
                vlogger::instance().console()->info(" SS result container cache : failed ");
            
            
        }
    }
}
#pragma GCC diagnostic pop
