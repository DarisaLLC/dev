#ifndef __QTIME_FRAMES_
#define __QTIME_FRAMES_

#ifdef __OCV_VIDEO_
#include "cvVideoPlayer.h"
#endif
#include <cinder/Channel.h>
#include <cinder/Area.h>
#include <limits>
#include "cinder/Thread.h"
#include "sshist.hpp"
#include "roiWindow.h"
#include <fstream>
#include <mutex>
#include <memory>
#include <map>
#include "core/timestamp.h"
#include "mediaInfo.h"
#include "timed_types.h"
#include "AVReader.hpp"
#include "lifFile.hpp"


using namespace ci;
using namespace std;



typedef std::pair<SurfaceRef, index_time_t> SurTiIndexRef_t;

class seqFrameContainer : public tiny_media_info, public std::enable_shared_from_this<seqFrameContainer>
{
public:
    typedef std::shared_ptr<seqFrameContainer> ref;
    typedef std::weak_ptr<seqFrameContainer> weak_ref;
    typedef std::vector<SurTiIndexRef_t> container_t;
    typedef typename container_t::size_type container_index_t;
    typedef std::map<int64_t, container_index_t > indexToContainer;
    
    typedef std::function<void (float)> progress_callback_t;
    
    template<typename S, class... Args>
    static std::shared_ptr<seqFrameContainer> create (const S &);

    void set_progress_callback(progress_callback_t pg = nullptr) const;
    
    // todo: also return index to time mapping 
    size_t convertTo (std::vector<roiWindow<P8U> >& dst);
    
    // Initializes for the movie. Frame indices are generated for unique increasing time stamps.
    // time-stamped Frames are copied and cached at the first load. Further references to the frame
    // by time stamp or index is from cache.
   
    bool isValid () const;
    
    tiny_media_info& media_info ();
    const std::ostream&  index_info (std::ostream& std_stream);
    const std::ostream& print_to_ (std::ostream& std_stream);

    const std::vector<std::string>& channel_names () const;
    void channel_names (const std::vector<std::string>& cnames) const; 
    
    // Load a frame at the time stamp indicated.
    // If a frame at that time stamp is already cached, it will return false
    // If the frame is loaded in, it will return true.
    bool loadFrame (const Surface8uRef frame, const time_spec_t& tic );
    
    // Check or get a frame at the time stamp indicated.
    const Surface8uRef  getFrame (const int64_t) const;
    const Surface8uRef  getFrame (const time_spec_t& ) const;
    bool checkFrame (const int64_t) const;
    bool checkFrame (const time_spec_t& ) const;
    
    int64_t indexFromTime (const time_spec_t& time) const;

    const index_time_t& currentIndexTime () const;
    size_t count () const;
    
    /** \brief returns the name of the concrete subclass.
     * \return the name of the concrete driver.
     */
    std::string getName () const;

    const timeToIndex_t& time2IndexMap () const { return m_tti; }
    const indexToTime_t& index2TimeMap () const { return m_itt; }
    
    
private:
    // Initializes for the movie. Frame indices are generated for unique increasing time stamps.
    // time-stamped Frames are copied and cached at the first load. Further references to the frame
    // by time stamp or index is from cache.
    seqFrameContainer ( const tiny_media_info& );
    seqFrameContainer ();
    
    bool loadFrame (const Surface8uRef, const index_time_t&);
    indexToContainer::const_iterator _checkFrame (const int64_t) const;
    indexToContainer::const_iterator _checkFrame (const time_spec_t&) const;
    bool make_unique_increasing_time_indices (const time_spec_t&, index_time_t&);
    mutable bool                mValid;
    mutable container_t      mFrames;
    mutable index_time_t     mCurrentIndexTime;
    timeToIndex_t m_tti;
    indexToTime_t m_itt;
    indexToContainer m_itIter;
    std::vector<time_spec_t> m_time_hist;
    mutable progress_callback_t m_progress_cb;
    mutable std::pair<uint32_t,uint32_t> m_stats;
    mutable std::vector<std::string> m_names;
    mutable std::mutex			mMutex;
};


class sequenceSample
{
public:
    sequenceSample (const seqFrameContainer::ref&, int x, int y);
    
private:
    seqFrameContainer::weak_ref m_weak_root;
    
    
    
};


#endif // __QTIME_VC_IMPL__
