//
//  algo_ssmt.cpp
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



const Rectf& ssmt_processor::measuredArea () const { return m_measured_area; }

/*
 * 1 monchrome channel. Compute 3D Standard Dev. per pixel
 */

void ssmt_processor::internal_find_moving_regions (std::vector<roiWindow<P8U>>& images){
    std::lock_guard<std::mutex> lock(m_mutex);
    generateVoxelsAndSelfSimilarities (images);

}


void ssmt_processor::find_moving_regions (const int channel_index){
    m_variance_peak_detection_done = false;
    volume_variance_peak_promotion(m_all_by_channel[channel_index]);
    while(!m_variance_peak_detection_done){ std::this_thread::yield();}
    m_instant_input = input_channel_selector_t(-1, channel_index);
    return internal_find_moving_regions(m_all_by_channel[channel_index]);
}


/**
 finalize_segmentation
    [grayscale,bi_level] -> create a blob runner, attach res_ready_lambda as callback, run blob runner
    [res_ready_lambda] processes only the top blob result (@todo reference blob result sorting criteria)
                       create an affine transform to affine warp the cell area
                       create a cell gl model for drawing it.
 
 @param mono gray scale image
 @param bi_level bi_level ( i.e. binarized image )
 ss_image_available signal calls this function
 signals geometry_ready with number of moving regions
 */
void ssmt_processor::finalize_segmentation (cv::Mat& mono, cv::Mat& bi_level){
    std::lock_guard<std::mutex> lock(m_segmentation_mutex);
    assert(mono.cols == bi_level.cols && mono.rows == bi_level.rows);
    assert(m_variance_peak_detection_done);
    vlogger::instance().console()->info("Locating moving regions");
    cv::Rect padded_rect, image_rect;

    // Sum variances under each cluster
    // Variances are in m_var_image
    
    // Creates Moving_regions from detected blobs in motion_energy space
    // Creates ssmt results from each.
    // @todo Add rules such as pick the one most centered to filter the results
    
    m_main_blob = labelBlob::create(mono,  bi_level, m_params.min_seqmentation_area(), 666);
    auto cache_path = mCurrentCachePath;
    m_regions.clear();
    
    std::function<labelBlob::results_cb> res_ready_lambda = [=](std::vector<blob>& blobs){

        std::string msg = toString(blobs.size());
        auto tid = std::this_thread::get_id();
        msg = toString(tid) + " Moving regions: " + msg;
        vlogger::instance().console()->info(msg);
        
        // @note: moving_region's id is the size of the container before it is added, i.e. sequential
        for (const blob& bb : blobs){
            m_regions.emplace_back(bb, (uint32_t(m_regions.size())));
        }
        int idx = 0;
        for (const moving_region& mr : m_regions){
            auto dis = shared_from_this();
            input_channel_selector_t inn (idx++,m_instant_input.channel());
            auto ref = ssmt_result::create (dis, mr,  inn);
            m_results.push_back(ref);
        }
        msg = toString(m_results.size());
        msg = " Results created: " + msg;
        vlogger::instance().console()->info(msg);
        int count = (int) m_regions.size();
        if (count != create_cache_paths()){
            std::string msg = " miscount in moving object path creation ";
            vlogger::instance().console()->error(msg);
        }

        
        if (signal_geometry_ready  && signal_geometry_ready->num_slots() > 0)
        {
            signal_geometry_ready->operator()(count, m_instant_input);
        }
     
    };

    
    if(bfs::exists(mCurrentCachePath)){
        std::string imagename = "voxel_binary_.png";
        auto image_path = cache_path / imagename;
        cv::imwrite(image_path.string(), bi_level);
        {
            std::string imagename = "peaks_.png";
            auto image_path = cache_path / imagename;
            cv::imwrite(image_path.string(), m_var_image);
        }
    }
    
    boost::signals2::connection results_ready_ = m_main_blob->registerCallback(res_ready_lambda);
    m_main_blob->run_async();
  
    
}



// Update. Called when cutoff offset has changed
void ssmt_processor::update (const input_channel_selector_t& in)
{
    std::weak_ptr<contractionLocator> weakCaPtr (in.isEntire() ? m_entireCaRef : m_results[in.region()]->locator());
    if (weakCaPtr.expired())
        return;

    auto caRef = weakCaPtr.lock();
    caRef->update ();
    median_leveled_pci(m_longterm_pci_tracksRef->at(0), in);
}


/*
    Affine Window Processing Implementation
*/

#if NOT_YET
void ssmt_processor::generate_affine_windows () {
    
    int channel_to_use = m_channel_count - 1;
    internal_generate_affine_windows(m_all_by_channel[channel_to_use]);
}

void ssmt_processor::internal_generate_affine_windows (const std::vector<roiWindow<P8U>>& rws){
    
    std::unique_lock<std::mutex> lock(m_mutex);

    auto affineCrop = [] (const vector<roiWindow<P8U> >::const_iterator& rw_src, cv::RotatedRect& rect){


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
            return cropped;
        };

        cvMatRefroiP8U(*rw_src, src, CV_8UC1);
        return matAffineCrop(src, rect);

    };


    auto horizontal_vertical_projections = [](const cv::Mat& mm){
        cv::Mat hz (1, mm.cols, CV_32F);
        cv::Mat vt (mm.rows, 1, CV_32F);
        reduce(mm,vt,1,CV_REDUCE_SUM,CV_32F);
        reduce(mm,hz,0,CV_REDUCE_SUM,CV_32F);
        std::vector<float> hz_vec(mm.cols);
        for (auto ii = 0; ii < mm.cols; ii++) hz_vec[ii] = hz.at<float>(0,ii);
        std::vector<float> vt_vec(mm.rows);
        for (auto ii = 0; ii < mm.rows; ii++) vt_vec[ii] = vt.at<float>(ii,0);
        return std::make_pair(hz_vec, vt_vec);
    };

    uint32_t count = 0;
    uint32_t total = static_cast<uint32_t>(rws.size());
    m_affine_windows.clear();
    m_hz_profiles.clear();
    m_vt_profiles.clear();
    vector<roiWindow<P8U> >::const_iterator vitr = rws.begin();
    do
    {
        cv::Mat affine = affineCrop(vitr, m_motion_mass);
        auto clone = affine.clone();
        auto hvpair = horizontal_vertical_projections(clone);
        m_affine_windows.push_back(clone);
        m_hz_profiles.push_back(hvpair.first);
        m_vt_profiles.push_back(hvpair.second);
        count++;
    }
    while (++vitr != rws.end());
    assert(count==total);

    internal_generate_affine_translations();
    internal_generate_affine_profiles();
    save_affine_windows ();
    save_affine_profiles();
    
}

void ssmt_processor::save_affine_windows (){
    
    if(bfs::exists(mCurrentSerieCachePath)){
        std::string subdir ("affine_cell_images");
        auto save_path = mCurrentSerieCachePath / subdir;
        boost::system::error_code ec;
        if(!bfs::exists(save_path)){
            bfs::create_directory(save_path, ec);
            switch( ec.value() ) {
                case boost::system::errc::success: {
                    std::string msg = "Created " + save_path.string() ;
                    vlogger::instance().console()->info(msg);
                }
                    break;
                default:
                    std::string msg = "Failed to create " + save_path.string() + ec.message();
                    vlogger::instance().console()->info(msg);
            }
        } // tried creatring it if was not already
        if(bfs::exists(save_path)){
            std::lock_guard<std::mutex> process_lock(m_io_mutex);
            auto writer = get_image_writer();
            writer->operator()(save_path.string(), m_affine_windows);
        }
    }
}


void ssmt_processor::save_affine_profiles (){
    
    if(bfs::exists(mCurrentSerieCachePath)){
        std::string subdir ("affine_cell_profiles");
        auto save_path = mCurrentSerieCachePath / subdir;
        boost::system::error_code ec;
        if(!bfs::exists(save_path)){
            bfs::create_directory(save_path, ec);
            switch( ec.value() ) {
                case boost::system::errc::success: {
                    std::string msg = "Created " + save_path.string() ;
                    vlogger::instance().console()->info(msg);
                }
                    break;
                default:
                    std::string msg = "Failed to create " + save_path.string() + ec.message();
                    vlogger::instance().console()->info(msg);
            }
        } // tried creatring it if was not already
        if(bfs::exists(save_path)){
            std::lock_guard<std::mutex> process_lock(m_io_mutex);
            auto this_csv_writer = std::make_shared<ioImageWriter>("hp");
            this_csv_writer->operator()(save_path.string(), m_hz_profiles);
            auto other_csv_writer = std::make_shared<ioImageWriter>("vp");
            other_csv_writer->operator()(save_path.string(), m_vt_profiles);
        }
    }
}

void ssmt_processor::internal_generate_affine_translations (){
    if(m_affine_windows.empty()) return;
    
    auto trans = [] (const cv::Mat& minus, const cv::Mat& plus){
        auto tw = minus.cols / 3;
        auto th = minus.rows / 3;
        auto width = minus.cols;
        auto height = minus.rows;
        auto cw = (width - tw)/2;
        auto ch = (height-th)/2;
        auto left = cv::Mat(minus, cv::Rect(tw,ch,tw,th));
        auto right = cv::Mat(minus, cv::Rect(cw,ch,tw,th));
        auto plus_left = cv::Mat(plus, cv::Rect(0,0,plus.cols/2, plus.rows));
        auto plus_right = cv::Mat(plus, cv::Rect(plus.cols/2,0,plus.cols/2, plus.rows));
        
        cv::Mat norm_xcorr_img;
        std::pair<cv::Point,cv::Point> max_loc;
        dPair max_val;
        cv::matchTemplate(plus_left,left, norm_xcorr_img, cv::TM_CCORR_NORMED, left);
        cv::minMaxLoc(norm_xcorr_img, NULL, &max_val.first, NULL, &max_loc.first);
        cv::matchTemplate(plus_right,right, norm_xcorr_img, cv::TM_CCORR_NORMED, right);
        cv::minMaxLoc(norm_xcorr_img, NULL, &max_val.second, NULL, &max_loc.second);
        fVector_2d lv(max_loc.first.x,max_loc.first.y);
        fVector_2d rv(max_loc.second.x+plus.cols/2.0f,max_loc.second.y);
        return lv.distance(rv);
    };
    
    std::vector<float> translations;
    auto d0 = trans(m_affine_windows[0],m_affine_windows[0]);
    translations.push_back(static_cast<float>(d0));
    auto iter = m_affine_windows.begin();
    while (++iter < m_affine_windows.end()){
        auto tt = trans(*iter,*(iter-1));
        translations.push_back(static_cast<float>(tt));
    }
    assert(translations.size() == m_affine_windows.size());
    m_affine_translations = translations;
    m_affine_translations[0] = m_affine_translations[1];
    
}
void ssmt_processor::internal_generate_affine_profiles(){
    
    m_norm_affine_translations = m_affine_translations;
    auto extremes = svl::norm_min_max(m_norm_affine_translations.begin(), m_norm_affine_translations.end());
    m_length_range.first = extremes.first;
    m_length_range.second = extremes.second;
    
 //   for (auto& mm : m_norm_affine_translations)
 //   {
//        m_cell_lengths.push_back(mm);
 //   }
}

#endif

#pragma GCC diagnostic pop


