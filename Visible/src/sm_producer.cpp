#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomma"


#include "opencv2/stitching.hpp" 
#include "sm_producer.h"
#include "sm_producer_impl.h"
#include "self_similarity.h"
#include "cinder_xchg.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <functional>

#include "core/simple_timing.hpp"
#include "vision/opencv_utils.hpp"
#include "vision/histo.h"



using namespace std::chrono;

using namespace boost;

namespace bfs=boost::filesystem;

using namespace ci;

using namespace svl;

namespace anonymous
{
    bool is_anaonymous_name (const boost::filesystem::path& pp, size_t check_size = 36)
    {
        string extension = pp.extension().string();
        return extension.length() == 0 && pp.filename().string().length() == check_size;
    }
    
    struct greaterScore {
        bool operator () (const sm_producer::outuple_t& left, const sm_producer::outuple_t& right)
        { // Score is at Index 1
            return std::get<1>(left) > std::get<1>(right);
        }
    };
    
    template <int index> struct TupleMore {
        template <typename Tuple>
        bool operator () (const Tuple& left, const Tuple& right) const {
            return std::get<index>(left) > std::get<index>(right); }
    };
    
    bool smatrix_ok(const sm_producer::sMatrix_t& sm, size_t d1)
    {
        bool ok = sm.size() == d1;
        if (!ok) return ok;
        for (int row = 0; row < d1; row++)
        {
            ok &= sm[row].size () == d1;
            if (!ok) return ok;
        }
        return ok;
    }
}

sm_producer::sm_producer ()
{
    _impl = std::shared_ptr<sm_producer::spImpl> (new sm_producer::spImpl);
}

bool sm_producer::load_content_file (const std::string& movie_fqfn)
{
    
    if (_impl) return _impl->loadMovie(movie_fqfn);
    return false;
}

bool sm_producer::load_image_directory (const std::string& dir_fqfn, sm_producer::sizeMappingOption szmap)
{
    if ( !boost::filesystem::exists( dir_fqfn ) ) return false;
    if ( !boost::filesystem::is_directory( dir_fqfn ) ) return false;
    
    
    if (_impl) return _impl->loadImageDirectory(dir_fqfn, szmap) > 0;
    return false;
}

std::future<bool> sm_producer::launch_async (int frames, const progress_fn_t& reporter)const {
    assert(has_content());
    
    auto fcnt = (frames == 0) ? _impl-> m_frameCount : frames;
    return std::async(std::launch::async, &sm_producer::spImpl::generate_ssm, _impl, fcnt, reporter);
}

bool sm_producer::operator() (int frames,  const progress_fn_t& reporter ) const
{
    if(has_content()){
        std::future<bool> bright = launch_async(frames, reporter);
        bright.wait();
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

bool sm_producer::has_content () const { return (frames_in_content() > 1 ); }

int sm_producer::frames_in_content() const { return (_impl) ? (int) _impl->frame_count(): -1; }


const sm_producer::sMatrix_t& sm_producer::similarityMatrix () const { return _impl->m_SMatrix; }

const sm_producer::sMatrixProjection_t& sm_producer::meanProjection (outputOrderOption ooo) const { assert(false); }

const sm_producer::sMatrixProjection_t& sm_producer::shannonProjection (outputOrderOption ooo) const { return _impl->m_entropies; }

std::vector<roiWindow<P8U>>& sm_producer::images () const
{
    return _impl->images ();
}

const std::vector< bfs::path >& sm_producer::paths () const
{
    return _impl->frame_paths();
}



const sm_producer::ordered_outuple_t& sm_producer::ordered_input_output (const sm_producer::outputOrderOption ooo) const
{
    return _impl->ordered_input_output (ooo);
}

const sm_producer::ordered_outuple_t& sm_producer::spImpl::ordered_input_output (const sm_producer::outputOrderOption ooo) const
{
    std::unique_lock <std::mutex> lock(m_mutex, std::try_to_lock);
    
    if (m_output_repo.empty())
    {
        static ordered_outuple_t default_empty_outuple;
        return default_empty_outuple;
    }
    
    auto output =  stl_utils::safe_get(m_output_repo, ooo, ordered_outuple_t ());
    
    // Not in the repo. Generate it if possible
    if (output.empty() && ! m_output_repo.empty() && ooo == sorted)
    {
        const ordered_outuple_t& canon = m_output_repo[input];
        // Copy the naturally ordered one: i.e. input
        output = canon;
        // Now sort them
        sort (output.begin(), output.end(), anonymous::TupleMore<1> ()) ;
        m_output_repo[sorted] = output;
    }
    
    return m_output_repo[ooo];
    
}

void sm_producer::spImpl::asset_reader_done_cb ()
{
    std::cout << " asset reader Done" << std::endl;
}

/*
 * Load all the frames
 * Could be producer / consumer for file fetching, Use shared memory, etc
 */
int sm_producer::spImpl::loadImageDirectory( const std::string& imageDir,  sm_producer::sizeMappingOption szmap, const std::vector<std::string>& supported_extensions)
{
    
    m_source_type = imageFileDirectory;
    
    paths_vector_t tmp_framePaths;
    
    // make a list of all image files in the directory
    // in to the tmp vector
    filesystem::directory_iterator end_itr;
    for( filesystem::directory_iterator i( imageDir ); i != end_itr; ++i )
    {
        std::unique_lock <std::mutex> lock(m_mutex, std::try_to_lock);
        // skip if not a file
        if( !filesystem::is_regular_file( i->status() ) ) continue;
        
        std::cout << "Checking " << i->path().string();
        
        if (std::find( supported_extensions.begin(), supported_extensions.end(), i->path().extension()  ) != supported_extensions.end())
        {
            std::cout << "  Extension matches " << std::endl;
            tmp_framePaths.push_back( i->path().string() );
        }
        else if (anonymous::is_anaonymous_name(i->path()))
        {
            std::cout << "  Anonymous rule matches " << std::endl;
            tmp_framePaths.push_back( i->path().string() );
        }
        else
        {
            std::cout << "Skipped " << i->path().string() << std::endl;
            continue;
        }
    }
    
    if (tmp_framePaths.empty()) return -1;
    
    m_framePaths.clear();
    m_loaded_ref.resize(0);
    
    std::cout << m_framePaths.size () << " Files "  << std::endl;
    
    // Paths and Image vectors will be
    // Get a map of the sizes
    std::map<iPair, int32_t> size_map;
    images_vector_t                 tmp_loaded_ref;
    std::vector<size_t>             tmp_paths_index;
    
    // Read data into memory buffer
    // Using tmp path vector
    for (boost::filesystem::path& pp : tmp_framePaths)
    {
        std::unique_lock<std::mutex> lock( m_mutex, std::try_to_lock );
        
        auto ipair = svl::image_io_read_surface (pp);
        roiWindow<P8U> rw;
        if (ipair.first)
        {
            rw = NewRedFromSurface(ipair.first);
        }
        else if (ipair.second)
        {
            rw = NewFromChannel(*ipair.second, 0);
        }
        else
        {
            std::cout << __FILE__ << " Unexpected error ";
            return -1;
        }
        
        size_map[rw.size()] +=1;
        
        tmp_loaded_ref.emplace_back(rw);
        
        std::cout << tmp_loaded_ref.size() << " Out of " << tmp_framePaths.size () << " Files "  << std::endl;
        
    }
    
    // All same size, or our pairwise compare function does not care
    if (size_map.size() == 1 || (size_map.size() > 1 &&  szmap == dontCare))
    {
        m_loaded_ref = tmp_loaded_ref;
        m_framePaths  = tmp_framePaths;
    }
    
    // All different size, expected to be the same, report by failing
    else if (size_map.size() > 1 &&  szmap == reportFail)
        return -1;
    
    // All different size, take the most common only
    else if (size_map.size() > 1 &&  szmap == mostCommon)
    {
        using pair_type = decltype(size_map)::value_type;
        auto pr = std::max_element
        (
         std::begin(size_map), std::end(size_map),
         [] (const pair_type & p1, const pair_type & p2) {
             return p1.second < p2.second;
         }
         );
        
        std::cout << "Size map contains " << size_map.size() << " most Common " << pr->first << std::endl;
        assert(tmp_framePaths.size() == tmp_loaded_ref.size());
        
        for (auto counter = 0; counter < tmp_loaded_ref.size(); counter++)
        {
            const roiWindow<P8U>& rr = tmp_loaded_ref[counter];
            const boost::filesystem::path& pp = tmp_framePaths[counter];
            if (rr.size() == pr->first)
            {
                m_loaded_ref.emplace_back(rr);
                m_framePaths.emplace_back(pp);
            }
        }
    }
    else
    {
        std::cout << __FILE__ << " Unexpected error ";
        return -1;
    }
    
    if (m_loaded_ref.empty()) return -1;
    
    m_frameCount = m_loaded_ref.size ();
    
    // Call the content loaded cb if any
    if (signal_content_loaded && signal_content_loaded->num_slots() > 0)
        signal_content_loaded->operator()();
    
    
    
    return (int) m_frameCount;
}

/*
 * Load all the frames in the movie
 */
int sm_producer::spImpl::loadMovie( const std::string& movieFile )
{
    std::unique_lock <std::mutex> lock(m_mutex);
    m_source_type = movie;
    
    {
        ScopeTimer timeit("avReader");
        m_assetReader = std::make_shared<avcc::avReader>(movieFile, false);
        m_assetReader->setUserDoneCallBack(std::bind(&sm_producer::spImpl::asset_reader_done_cb, this));
    }
    
    bool m_valid = m_assetReader->isValid();
    int rtn_val = -1;
    if (m_valid)
    {
        m_assetReader->info().printout();
        m_assetReader->run();
        m_qtime_cache_ref = seqFrameContainer::create (m_assetReader);
        
        // Now load every frame, convert and update vector
        m_loaded_ref.resize(0);
        {
            ScopeTimer timeit("convertTo");
            m_qtime_cache_ref->convertTo (m_loaded_ref);
        }
        m_frameCount = m_loaded_ref.size ();
        
        rtn_val = (int) m_frameCount;
    }
    
    // Call the content loaded cb if any
    if (signal_content_loaded && signal_content_loaded->num_slots() > 0)
        signal_content_loaded->operator()();
    
    return rtn_val;
    
}

void sm_producer::spImpl::loadImages (const images_vector_t& images)
{
    std::unique_lock <std::mutex> lock(m_mutex);
    
    m_source_type = imageInMemory;
    m_loaded_ref.resize(0);
    vector<roiWindow<P8U> >::const_iterator vitr = images.begin();
    do
    {
        m_loaded_ref.push_back (*vitr++);
        // Call the content loaded cb if any
        int count = static_cast<int>(m_loaded_ref.size());
        double done_pc = (100.0 * count)/static_cast<double>(images.size());
        if (signal_frame_loaded  && signal_frame_loaded->num_slots() > 0)
            signal_frame_loaded->operator()(count, done_pc);
    }
    while (vitr != images.end());
    m_frameCount = m_loaded_ref.size ();
    
    // Call the content loaded cb if any
    if (signal_content_loaded && signal_content_loaded->num_slots() > 0)
        signal_content_loaded->operator()();
    
}

bool sm_producer::spImpl::done_grabbing () const
{
    // unique lock. forces new shared locks to wait untill this lock is release
    std::unique_lock<std::mutex> lock( m_mutex, std::try_to_lock );
    
    return m_frameCount != 0 && m_qtime_cache_ref->count() == m_frameCount;
}

bool  sm_producer::spImpl::image_file_entropy_result_ok () const
{
    assert (type() == imageFileDirectory);
    bool ok = m_frameCount != 0 && !m_entropies.empty();
    ok = m_loaded_ref.size() == m_framePaths.size();
    ok &= m_entropies.size() == m_loaded_ref.size();
    return ok;
}

bool sm_producer::spImpl::setup_image_directory_result_repo () const
{
    if (! image_file_entropy_result_ok()) return false;
    
    // Setup new results results tuple
    ordered_outuple_t input_result;
    for (size_t rr = 0; rr < m_loaded_ref.size(); rr++)
    {
        outuple_t ot (rr, m_entropies[rr], m_framePaths[rr], m_loaded_ref[rr]);
        input_result.push_back(ot);
    }
    m_output_repo[outputOrderOption::input] = input_result;
    return true;
    
}


// @todo add sampling and offset
// @todo direct shortterm generation. 
bool sm_producer::spImpl::generate_ssm ( int frames, const sm_producer::progress_fn_t& reporter)
{
    std::unique_lock<std::mutex> lock( m_mutex, std::try_to_lock );
    
    // Get a new similarity engine
    // Note: get execution times with   svl::stats<float>::PrintTo(simi->timeStats(), & std::cout);
    self_similarity_producerRef simi = std::make_shared<self_similarity_producer<P8U> > (frames, 0, reporter);
    
    // Invalidate last results map
    m_output_repo.clear();

    // This is a blocking call
    simi->fill(m_loaded_ref);
    
    m_entropies.resize (0);
    m_SMatrix.resize (0);

    bool ok = simi->entropies (m_entropies);
    // Fetch the SS matrix and verify
    simi->selfSimilarityMatrix(m_SMatrix);
    ok = ok && anonymous::smatrix_ok(m_SMatrix, m_entropies.size());
    
    
    if (!ok){
        std::cout << " ERR OR " << std::endl;
        return false;
    }
    
    // Call the sm 1d ready cb if any
    if (ok && signal_sm1d_available && signal_sm1d_available->num_slots() > 0)
        signal_sm1d_available->operator()();
    
    // Call the sm 2d ready cb if any
    if (ok && signal_sm2d_available && signal_sm2d_available->num_slots() > 0)
        signal_sm2d_available->operator()();
    
    if (type() == imageFileDirectory)
        ok &= setup_image_directory_result_repo();
    
    return ok;
}



template boost::signals2::connection sm_producer::registerCallback(const std::function<sm_producer::sig_cb_content_loaded>&);
template boost::signals2::connection sm_producer::registerCallback(const std::function<sm_producer::sig_cb_frame_loaded>&);


#pragma GCC diagnostic pop

