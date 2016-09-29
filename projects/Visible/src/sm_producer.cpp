#include "sm_producer.h"
#include "sm_producer_impl.h"
#include "self_similarity.h"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <functional>
#include "app_utils.hpp"
#include "cinder/qtime/Quicktime.h"
#include "cinder_xchg.hpp"
#include "core/simple_timing.hpp"

#include "qtimeAvfLink.h"

using namespace std::chrono;

using namespace boost;


sm_producer::sm_producer ()
{
    
    _impl = std::shared_ptr<sm_producer::spImpl> (new sm_producer::spImpl);
    
}


bool sm_producer::load_content_file (const std::string& movie_fqfn)
{

    if (_impl) return _impl->loadMovie(movie_fqfn);
    return false;
}

bool sm_producer::operator() (int start_frame, int frames ) const
{
    if (_impl)
    {
        std::future<bool> bright = std::async(std::launch::deferred, &sm_producer::spImpl::generate_ssm, _impl, start_frame, frames);
        return bright.get();
    }
    return false;
}

void sm_producer::load_images(const images_vector_t &images)
{
    if (_impl) _impl->loadImages (images);
}

template<typename T> boost::signals2::connection
sm_producer::registerCallback (const std::function<T> & callback)
{
    return _impl->registerCallback  (callback);
}

template<typename T> bool
sm_producer::providesCallback () const
{
    return _impl->providesCallback<T> ();
}

bool sm_producer::set_auto_run_on() const { return (_impl) ? _impl->set_auto_run_on() : false; }
bool sm_producer::set_auto_run_off() const { return (_impl) ? _impl->set_auto_run_off() : false; }
int sm_producer::frames_in_content() const { return (_impl) ? (int) _impl->frame_count(): -1; }

//grabber_error sm_producer::last_error () const { if (_impl) return _impl->last_error(); return grabber_error::Unknown; }

const sm_producer::sMatrix_t& sm_producer::similarityMatrix () const { return _impl->m_SMatrix; }

//const sMatrixProjection_t& sm_producer::meanProjection () const;

const sm_producer::sMatrixProjection_t& sm_producer::shannonProjection () const { return _impl->m_entropies; }


/*
 * Load all the frames
 */
int sm_producer::spImpl::loadMovie( const std::string& movieFile )
{
    std::unique_lock <std::mutex> lock(m_mutex);

    {
        ScopeTimer timeit("avReader");
        m_assetReader = std::make_shared<avcc::avReader>(movieFile, false);
    }
    
    bool m_valid = m_assetReader->isValid();
    if (m_valid)
    {
        m_assetReader->info().printout();
        m_assetReader->run();
        m_qtime_cache_ref = qTimeFrameCache::create (m_assetReader);
        
        // Now load every frame, convert and update vector
        m_loaded_ref.resize(0);
        {
            ScopeTimer timeit("convertTo");
            m_qtime_cache_ref->convertTo (m_loaded_ref);
        }
        _frameCount = m_loaded_ref.size ();
        if (m_auto_run) generate_ssm (0,0);
        return (int) _frameCount;
        
    }
    return -1;

}

void sm_producer::spImpl::loadImages (const images_vector_t& images)
{
    m_loaded_ref.resize(0);
    vector<roiWindow<P8U> >::const_iterator vitr = images.begin();
    do
    {
        m_loaded_ref.push_back (*vitr++);
    }
    while (vitr != images.end());
    _frameCount = m_loaded_ref.size ();
    if (m_auto_run) generate_ssm (0,0);
        
}

bool sm_producer::spImpl::done_grabbing () const
{
    // unique lock. forces new shared locks to wait untill this lock is release
//    std::lock_guard <std::mutex> lock(m_mutex);
    return _frameCount != 0 && m_qtime_cache_ref->count() == _frameCount;
}



// @todo add sampling and offset
bool sm_producer::spImpl::generate_ssm (int start_frames, int frames)
{
static    double tiny = 1e-10;
    
    self_similarity_producerRef simi = std::make_shared<self_similarity_producer<P8U> > (frames, 0);

    
    simi->fill(m_loaded_ref);
    m_entropies.resize (0);
    bool ok = simi->entropies (m_entropies);
    m_SMatrix.resize (0);
    simi->selfSimilarityMatrix(m_SMatrix);
    std::cout << simi->matrixSz() << std::endl;
    svl::stats<float>::PrintTo(simi->timeStats(), & std::cout);

    
    return ok;
}


//
//// Generic method to load frames from a rcFrameGrabber
//int sm_producer::spImpl::saveFrames(std::string imageExportDir) const
//{
//    if (imageExportDir.empty()) return 0;
//
//    images_vector_t::const_iterator imgItr = _fileImages.begin ();
//
//    time_spec_t duration = time_spec_t::now();
//
//    int32 i = 0;
//    for (; imgItr != _fileImages.end(); imgItr++, i++)
//    {
//        std::ostringstream oss;
//        oss << imageExportDir << "/" << "image" << setfill ('0') << setw(4) << i << ".jpg";
//        std::string fn (oss.str ());
//        vf_utils::ci2rc2ci::ImageExport2JPG ( *imgItr, fn);
//        fn = std::string ("chmod 644 ") + fn;
//        ::system( fn.c_str() );
//
//    }
//
//    // Done. Report
//    duration = time_spec_t::now() - duration;
//    cout <<  endl << i << " Images Exported in " << duration.secs() << "Seconds" << endl;
//
//    return i;
//}



template boost::signals2::connection sm_producer::registerCallback(const std::function<sm_producer::sig_cb_content_loaded>&);
template boost::signals2::connection sm_producer::registerCallback(const std::function<sm_producer::sig_cb_frame_loaded>&);

