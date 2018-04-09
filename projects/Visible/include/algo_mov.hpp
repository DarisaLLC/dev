#ifndef __ALGO_MOV__
#define __ALGO_MOV__


#include <map>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>
#include <sstream>
#include <typeindex>
#include <map>
#include <future>
#include "core/singleton.hpp"
#include "async_producer.h"
#include "signaler.h"
#include "sm_producer.h"
#include "cinder_xchg.hpp"
#include "vision/histo.h"
#include "vision/opencv_utils.hpp"
#include "opencv2/video/tracking.hpp"
#include "getLuminanceAlgo.hpp"
using namespace cv;

// For base classing
class movSignaler : public base_signaler
{
    virtual std::string
    getName () const { return "movSignaler"; }
};

class mov_processor : public movSignaler
{
public:
    
    typedef void (mov_cb_content_loaded) ();
    typedef void (mov_cb_frame_loaded) (int&, double&);
    typedef void (mov_cb_sm1d_available) (int&);
    typedef void (mov_cb_sm1dmed_available) (int&,int&);
    typedef std::vector<roiWindow<P8U>> channel_images_t;
    
    mov_processor ()
    {
        signal_content_loaded = createSignal<mov_processor::mov_cb_content_loaded>();
        signal_frame_loaded = createSignal<mov_processor::mov_cb_frame_loaded>();
        signal_sm1d_available = createSignal<mov_processor::mov_cb_sm1d_available>();
        signal_sm1dmed_available = createSignal<mov_processor::mov_cb_sm1dmed_available>();
        m_sm = std::shared_ptr<sm_producer> ( new sm_producer () );
    }
    
  //  const smProducerRef sm () const { return m_sm; }
//    std::shared_ptr<sm_filter> smFilterRef () { return m_smfilterRef; }
    
protected:
    boost::signals2::signal<mov_processor::mov_cb_content_loaded>* signal_content_loaded;
    boost::signals2::signal<mov_processor::mov_cb_frame_loaded>* signal_frame_loaded;
    boost::signals2::signal<mov_processor::mov_cb_sm1d_available>* signal_sm1d_available;
    boost::signals2::signal<mov_processor::mov_cb_sm1dmed_available>* signal_sm1dmed_available;
    
private:
    
    // Assumes RGB
    void load_channels_from_images (const std::shared_ptr<qTimeFrameCache>& frames)
    {
        int64_t fn = 0;
        m_channel_images.clear();
        m_channel_images.resize (3);
        cv::Mat mat;
        while (frames->checkFrame(fn))
        {
            auto su8 = frames->getFrame(fn++);
            auto m1 = svl::NewRedFromSurface(su8);
            m_channel_images[0].emplace_back(m1);
            auto m2 = svl::NewGreenFromSurface(su8);
            m_channel_images[1].emplace_back(m2);
            auto m3 = svl::NewBlueFromSurface(su8);
            m_channel_images[2].emplace_back(m3);
        }
    }
    
 
    
    // Note tracks contained timed data.
    void entropiesToTracks (namedTrackOfdouble_t& track)
    {
        track.second.clear();
        if (m_smfilterRef->median_levelset_similarities())
        {
            // Signal we are done with median level set
            static int dummy;
            if (signal_sm1dmed_available && signal_sm1dmed_available->num_slots() > 0)
                signal_sm1dmed_available->operator()(dummy, dummy);
        }

        auto bee = m_smfilterRef->entropies().begin();
        auto mee = m_smfilterRef->median_adjusted().begin();
        for (auto ss = 0; bee != m_smfilterRef->entropies().end() && ss < frame_count(); ss++, bee++, mee++)
        {
            index_time_t ti;
            ti.first = ss;
            timed_double_t res;
            res.first = ti;
            res.second = *mee;
            track.second.emplace_back(res);
        }
    }
    
    size_t frame_count () const
    {
        if (m_channel_images[0].size() == m_channel_images[1].size() && m_channel_images[1].size() == m_channel_images[2].size())
            return m_channel_images[0].size();
        else return 0;
    }
    
    void create_named_tracks (const std::vector<std::string>& names)
    {
        m_tracksRef = std::make_shared<vectorOfnamedTrackOfdouble_t> ();
        
        m_tracksRef->resize (names.size ());
        for (auto tt = 0; tt < names.size(); tt++)
            m_tracksRef->at(tt).first = names[tt];
    }

    void compute_channel_statistics_threaded ()
    {
        std::vector<timed_double_vec_t> cts (3);
        std::vector<std::thread> threads(3);
        for (auto tt = 0; tt < 3; tt++)
        {
            threads[tt] = std::thread(IntensityStatisticsRunner(),
                                      std::ref(m_channel_images[tt]), std::ref(cts[tt]));
        }
        std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

        for (auto tt = 0; tt < 3; tt++)
            m_tracksRef->at(tt).second = cts[tt];
    }
    

    
    void loadImagesToMats (const sm_producer::images_vector_t& images, std::vector<cv::Mat>& mts)
    {
        mts.resize(0);
        vector<roiWindow<P8U> >::const_iterator vitr = images.begin();
        do
        {
            mts.emplace_back(vitr->height(), vitr->width(), CV_8UC(1), vitr->pelPointer(0,0), size_t(vitr->rowUpdate()));
        }
        while (++vitr != images.end());
    }
    
public:
    // Run to get Entropies and Median Level Set
    std::shared_ptr<vectorOfnamedTrackOfdouble_t>  run (const std::shared_ptr<qTimeFrameCache>& frames, const std::vector<std::string>& names,
                                                bool test_data = false)
    {
        
        load_channels_from_images(frames);
        create_named_tracks(names);
        compute_channel_statistics_threaded();

//        channel_images_t c2 = m_channel_images[2];
 //       loadImagesToMats(c2, m_channel2_mats);


     //   auto oflow = compute_oflow_threaded ();
        

//        auto sp =  sm();
//        sp->load_images (c2);
//
//        // Call the content loaded cb if any
//        if (signal_content_loaded && signal_content_loaded->num_slots() > 0)
//            signal_content_loaded->operator()();
//
//        std::packaged_task<bool()> task([sp](){ return sp->operator()(0, 0);}); // wrap the function
//        std::future<bool>  future_ss = task.get_future();  // get a future
//        std::thread(std::move(task)).detach(); // launch on a thread
//        if (future_ss.get())
//        {
//            // Signal we are done with ACI
//            static int dummy;
//            if (signal_sm1d_available && signal_sm1d_available->num_slots() > 0)
//                signal_sm1d_available->operator()(dummy);
//
//            const deque<double>& entropies = sp->shannonProjection ();
//            const deque<deque<double>>& smat = sp->similarityMatrix();
//            m_smfilterRef = std::make_shared<sm_filter> (entropies, smat);
//
//
//            update ();
//        }
        return m_tracksRef;
    }
    
    const namedTrackOfdouble_t& similarity_track () const { return m_tracksRef->at(2); }
    
    // Update. Called also when cutoff offset has changed
    void update ()
    {
//        if(m_tracksRef && !m_tracksRef->empty() && m_smfilterRef && m_smfilterRef->isValid())
//            entropiesToTracks(m_tracksRef->at(2));
    }
    
private:
    smProducerRef m_sm;
    channel_images_t m_images;
    std::vector<channel_images_t> m_channel_images;
    std::vector<cv::Mat> m_channel2_mats;
    
    Rectf m_all;
    std::shared_ptr<sm_filter> m_smfilterRef;
    std::deque<double> m_medianLevel;
    std::shared_ptr<vectorOfnamedTrackOfdouble_t>  m_tracksRef;
};


#endif