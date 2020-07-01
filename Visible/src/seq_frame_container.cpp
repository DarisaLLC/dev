

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomma"

#include "seq_frame_container.hpp"
#include <iterator>
#include "core/stl_utils.hpp"
#include "cinder_xchg.hpp"
#include <algorithm>
#include "cinder/ip/Fill.h"
#include "vision/roiMultiWindow.h"


/*
 *  Concepte:
 *  Movie consisting of M frames identified by time and index in the movie context
 *  seq_frame_container is a container of frames identified by time and index
 *                access is done through the following maps
 *                movie time <-> movie index
 *                movie index -> container index ( or iterator )
 *
 *  If the container is large enough for all frames in the movie, after initial load, all frame fetches are nearly free
 *  To Do: If the container is smaller then number of frames in the movie, a least used stack will be used to refresh
 */

using namespace ci;
using namespace ci::ip;
using namespace stl_utils;

#define OCV_PLAYER

std::string seqFrameContainer::getName () const { return "seqFrameContainer"; }

template<>
std::shared_ptr<seqFrameContainer> seqFrameContainer::create (const lifIO::LifSerie& lifserie)
{
    std::vector<std::string> names { "green", "red", "gray" };
    
    tiny_media_info tm;
    tm.count = (uint32_t)lifserie.getNbTimeSteps();
    auto dims = lifserie.getSpatialDimensions();
    tm.duration = lifserie.frame_duration_ms();
    tm.mFps = lifserie.frame_rate();
    tm.mIsLifSerie = true;
    tm.mIsImageFolder = false;
    tm.mChannels = (uint32_t)lifserie.getChannels().size();
    
    tm.channel_size.width = static_cast<int>(dims[0]);
    tm.channel_size.height = static_cast<int>(dims[1]);
    // Buffer 2D size. See lif_serie_data
    tm.size = tm.channel_size;
    tm.size.height *= tm.mChannels;
    cm_time frame_time;
    
    seqFrameContainer::ref thisref (new seqFrameContainer(tm));
    
    thisref->channel_names (names);
    
    if (lifserie.getDurations().size ())
    {
        if (tm.count > 1)
        {
            std::vector<lifIO::LifSerieHeader::timestamp_t>::const_iterator tItr = lifserie.getTimestamps().begin();
            
            int inc = 0;
            for (auto frame_count = 0; frame_count < tm.count; frame_count+=inc)
            {
                Channel8u frame (tm.getWidth(), tm.getHeight());
                lifserie.fill2DBuffer(frame.getData(), frame_count);
                Surface8uRef chsurface = Surface8u::create(frame);
                
                if (chsurface){
                    bool check = thisref->loadFrame(chsurface, frame_time);
                    if (! check){
                        std::cout << std::endl << "-----------------" << frame_count << "--------------" << std::endl;
                    }
                    
                    // Increment durations by number of channels
                    cm_time ts((*tItr)/(10000.0));
                    frame_time = frame_time + ts;
                    inc = 1;
                    
                    // Step the durations number of channel times.
                    tItr += tm.mChannels;
                }
                else{
                    inc = 0;
                    std::cout << " Error after  frame count " << frame_count << std::endl;
                }
            }
            
            thisref->mValid = tm.count == thisref->count();
            
            return thisref;
        }
        else if (tm.count == 1) // @todo
        {
            return thisref;
        }
    }
    return thisref;
}

#ifdef __OCV_VIDEO_
template<>
std::shared_ptr<seqFrameContainer> seqFrameContainer::create (const cvVideoPlayer::ref& mMovie)
{
    tiny_media_info minfo;
    minfo.size.width = mMovie->getSize().x;
    minfo.size.height = mMovie->getSize().y;
    minfo.mFps = mMovie->getFrameRate();
    minfo.count = mMovie->getNumFrames ();
    minfo.duration = mMovie->getDuration();
    minfo.mChannels = 4;
    minfo.channel_size = minfo.size;
    minfo.mIsMovie = true;
    
    seqFrameContainer::ref thisref (new seqFrameContainer(minfo));
    
    for(auto ff = 0; ff < minfo.count; ff++){
        auto sframe = mMovie->createSurface();
        auto ftime = ff / minfo.mFps;
        cm_time ft (ftime);
        bool check = thisref->loadFrame(sframe, ft);
        if (! check){
            std::cout << std::endl << "-----------------" << ff << "--------------" << std::endl;
        }
    }
    thisref->mValid = thisref->count() == minfo.count;
    return thisref;
    
}
#endif

template<>
std::shared_ptr<seqFrameContainer> seqFrameContainer::create (const std::shared_ptr<avcc::avReader>& asset_reader)
{
    seqFrameContainer::ref thisref (new seqFrameContainer(asset_reader->info()));
    while (! asset_reader->isEmpty())
    {
        cm_time ts;
        Surface8uRef frame;
        asset_reader->pop(ts, frame);
        thisref->loadFrame(frame, time_spec_t(ts));
    }
    return thisref;
}

template<>
std::shared_ptr<seqFrameContainer> seqFrameContainer::create (const std::vector<ci::Surface8uRef>& folderImages)
{
    tiny_media_info minfo;
    minfo.mFps = 1.0;
    minfo.count = static_cast<uint32_t>(folderImages.size());
    minfo.duration = 1.0;
    seqFrameContainer::ref thisref (new seqFrameContainer(minfo));
    cm_time ts;
    cm_time duration (minfo.duration);
    for (auto iitr = folderImages.begin(); iitr < folderImages.end(); iitr++)
    {
        thisref->loadFrame(*iitr, time_spec_t(ts));
        ts = ts + duration;
    }
    return thisref;
}

seqFrameContainer::seqFrameContainer () : mValid(false)
{
    m_time_hist.resize (0);
    m_stats.first = m_stats.second = 0;
}

seqFrameContainer::seqFrameContainer ( const tiny_media_info& info ) : tiny_media_info(info)
{
    m_time_hist.resize (0);
    m_stats.first = m_stats.second = 0;
}

/*
 * Get the frame at offset from current.
 * Updates current only if successful and consistent
 * Returns a shared_ptr to frame. Valid until that frame is still alive.
 */

const Surface8uRef  seqFrameContainer::getFrame (int64_t offset) const
{
    std::unique_lock<std::mutex> lock( mMutex , std::try_to_lock);
    Surface8uRef s8;
    indexToContainer::const_iterator p = _checkFrame(offset);
    if (p == m_itIter.end()) return s8;
    mCurrentIndexTime.first = p->second - 1;
    mCurrentIndexTime.second = m_itt.at(p->second-1);
    assert(p->second > 0 && p->second <= mFrames.size());
    
    return mFrames[p->second-1].first;
}

/*
 * Get the frame at a time
 * Updates current only if successful and consistent
 * Returns a shared_ptr to frame. Valid until that frame is still alive.
 */

const Surface8uRef seqFrameContainer::getFrame(const time_spec_t& dtime) const
{
    std::unique_lock<std::mutex> lock( mMutex, std::try_to_lock );
    Surface8uRef s8;
    indexToContainer::const_iterator fc = _checkFrame (dtime);
    if (fc == m_itIter.end())
        return s8;
    
    assert(fc->second > 0 && fc->second <= mFrames.size());
    mCurrentIndexTime.first = fc->second -  1;
    mCurrentIndexTime.second = dtime;
    return mFrames[fc->second-1].first;
    
}

size_t seqFrameContainer::count () const { return mFrames.size(); }

bool seqFrameContainer::checkFrame (const time_spec_t& dtime) const
{
    return _checkFrame(dtime) != m_itIter.end();
}


bool seqFrameContainer::checkFrame (int64_t offset) const
{
    return _checkFrame(offset) != m_itIter.end();
}

bool seqFrameContainer::isValid () const
{
    if (mValid) return !mFrames.empty();
    return mValid;
}

tiny_media_info& seqFrameContainer::media_info ()
{
    return *((tiny_media_info*) this);
}

void seqFrameContainer::set_progress_callback(const progress_callback_t pg) const{
    if(pg != nullptr){
        m_progress_cb = std::bind(pg, placeholders::_1);
    }
    else{
        m_progress_cb = nullptr;
    }
}
const std::ostream& seqFrameContainer::print_to_ (std::ostream& std_stream)
{
    std_stream << (tiny_media_info*)this << std::endl;
    std_stream << "Hits:    " << m_stats.first << std::endl;
    std_stream << "Misses: " << m_stats.second << std::endl;
    return std_stream;
}

const std::ostream&  seqFrameContainer::index_info (std::ostream& std_stream){
    std_stream << m_itIter;
    return std_stream;
}

/*
 * Increment the time histogram if the new time stamp is increasing.
 */
bool seqFrameContainer::make_unique_increasing_time_indices (const time_spec_t& tic, index_time_t& outi)
{
    bool check = m_time_hist.empty() || tic > m_time_hist.back();
    if (!check) return check;
    auto pre_size = m_time_hist.size();
    outi = make_pair((int64_t)m_time_hist.size(), tic);
    m_time_hist.push_back(tic);
    return (m_time_hist.size() == (pre_size+1) && outi.second == tic);
}


bool seqFrameContainer::loadFrame (const Surface8uRef frame, const index_time_t& tid )
{
    std::unique_lock <std::mutex> lock( mMutex , std::try_to_lock );
    
    assert (frame );
    index_time_t ltid (tid);
    //    SurTiIndexRef_t pp (std::make_shared<Surface8u>(frame->clone(true)), ltid);
    mFrames.emplace_back(std::make_shared<Surface8u>(frame->clone(true)), ltid);
    m_tti[tid.second] = tid.first;
    m_itt[tid.first] = tid.second;
    m_itIter[tid.first] = mFrames.size();
    m_stats.first += 1;
    return true;
}

/*
 * Return true of frame was loaded, false if it was already in cache
 */

bool seqFrameContainer::loadFrame (const Surface8uRef frame, const time_spec_t& tic )
{
    std::unique_lock <std::mutex> lock( mMutex , std::try_to_lock );
    assert(frame );
    if (!mFrames.empty())
    {
        // Check to see if it exists
        auto pp = _checkFrame (tic);
        if (pp != m_itIter.end()) return false;
    }
    // It is not in the cache.
    // Get a time_index for it and load it
    index_time_t ti;
    shared_from_this()->make_unique_increasing_time_indices (tic, ti);
    return loadFrame(frame, ti);
}

/*
 * Returns an iterator to the container of <TimeIndex,SurfaceRef>
 * The end iterator is returned if entry is not found.
 */

int64_t seqFrameContainer::indexFromTime (const time_spec_t& time) const
{
    std::unique_lock<std::mutex> lock( mMutex, std::try_to_lock );
    
    indexToContainer::const_iterator ci = _checkFrame(time);
    if(ci == m_itIter.end()) return -1;
    if(ci == m_itIter.begin()) return 0;
    return std::distance(m_itIter.begin(), ci);
}

const index_time_t& seqFrameContainer::currentIndexTime () const
{
    return mCurrentIndexTime;
}

seqFrameContainer::indexToContainer::const_iterator seqFrameContainer::_checkFrame (const time_spec_t& query_time) const
{
    std::unique_lock<std::mutex> lock( mMutex, std::try_to_lock );
    
    // First see if this time exist in time index maps
    timeToIndex_t::const_iterator low, prev;
    low = m_tti.lower_bound(query_time);
    if (low == m_tti.end()) {
        return m_itIter.end();
    }
    m_stats.second += 1;
    
    if (low == m_tti.begin())
        return m_itIter.begin();
    
    prev = low;
    --prev;
    if ((query_time - prev->first) < (low->first - query_time))
        return m_itIter.find(prev->second);
    else
        return m_itIter.find(low->second);
    
    
}

seqFrameContainer::indexToContainer::const_iterator seqFrameContainer::_checkFrame (const int64_t query_index) const
{
    return m_itIter.find(query_index);
}

size_t seqFrameContainer::convertTo(std::vector<roiWindow<P8U> >& dst)
{
    dst.clear ();
    
    container_t::const_iterator  st = mFrames.begin();
    for (; st < mFrames.end(); st++)
    {
        roiWindow<P8U> rp = svl::NewGrayFromSurface(st->first);
        dst.push_back(rp);
    }
    return dst.size();
}

const std::vector<std::string>& seqFrameContainer::channel_names () const { return m_names; }
void seqFrameContainer::channel_names (const std::vector<std::string>& cnames) const { m_names = cnames; }

#pragma GCC diagnostic pop


#if 0
bool getSurfaceAndCopy (cached_frame_ref& ptr)
{
    std::lock_guard<std::mutex> lk(mx);
    if (!mMovie) return false;
    Surface8u surf = mMovie->getSurface ();
    if (!surf) return false;
    
    Surface::Iter iter = mSurface->getIter ( surf->getBounds() );
    int rows = 0;
    while (iter.line () )
    {
        uint8_t* pels = ptr->rowPointer (rows++);
        while ( iter.pixel () ) *pels++ = iter.g ();
    }
    return true;
}
#endif
