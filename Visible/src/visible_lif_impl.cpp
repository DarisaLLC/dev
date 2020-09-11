//
//  visible_lif_impl.cpp
//  Visible
//
//  Created by Arman Garakani on 5/13/16.
//
//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-private-field"
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#pragma GCC diagnostic ignored "-Wcomma"

#include "opencv2/stitching.hpp"
#include <strstream>
#include <algorithm>
#include <future>
#include <mutex>
#include <stdio.h>
#include "guiContext.h"
#include "LifContext.h"

#include "cinder/gl/gl.h"
#include "cinder/Timer.h"
#include "cinder/ImageIo.h"
#include "cinder/ip/Blend.h"
#include "opencv2/highgui.hpp"
#include "cinder/ip/Flip.h"
#include "otherIO/lifFile.hpp"

#include "timed_types.h"
#include "cinder_xchg.hpp"
#include "visible_layout.hpp"
#include "vision/opencv_utils.hpp"
#include "core/stl_utils.hpp"
#include "lif_content.hpp"
#include "core/signaler.h"
#include "core/core.hpp"
#include "contraction.hpp"
#include "logger/logger.hpp"
#include "cinder/Log.h"
#include "CinderImGui.h"
#include "ImGuiExtensions.h" // for 64bit count support
#include "Resources.h"
#include "cinder_opencv.h"
#include "imguivariouscontrols.h"
#include <boost/range/irange.hpp>
#include "core/stl_utils.hpp"
#include "imgui_visible_widgets.hpp"
//#include "imguifilesystem.h"

using namespace ci;
using namespace ci::app;
using namespace std;
using namespace svl;
using namespace boost;
namespace bfs=boost::filesystem;
namespace ui=ImGui;


// #define SHORTTERM_ON

#define ONCE(x) { static int __once = 1; if (__once) {x;} __once = 0; }

namespace {
    std::map<std::string, unsigned int> named_colors = {{"Red", 0xFF0000FF }, {"red",0xFF0000FF },{"Green", 0xFF00FF00},
        {"green", 0xFF00FF00},{ "PCI", 0xFFFF0000}, { "pci", 0xFFFF0000}, { "Short", 0xFFD66D3E}, { "short", 0xFFFFFF3E},
        { "Synth", 0xFFFB4551}, { "synth", 0xFFFB4551},{ "Length", 0xFFFFFF3E}, { "length", 0xFFFFFF3E},
        { "Force", 0xFFFB4551}, { "force", 0xFFFB4551}, { "Elongation", 0xFF00FF00}, { "elongation", 0xFF00FF00}
    };
    std::map<unsigned int, unsigned int> numbered_half_colors = {{0, 0x330000FF },{1, 0x3300FF00},{2, 0x33FF0000} };

    struct ScopedWindowWithFlag : public ci::Noncopyable {
        ScopedWindowWithFlag( const char* label, bool* p_open = nullptr, int windowflags = 0 );
        ~ScopedWindowWithFlag();
    };

    ScopedWindowWithFlag::ScopedWindowWithFlag( const char* label, bool* p_open, int window_flags )
    {
        ImGui::Begin( label, p_open, window_flags );
    }
    ScopedWindowWithFlag::~ScopedWindowWithFlag()
    {
        ImGui::End();
    }

    bool getPosSizeFromWindow(const char* last_window, ImVec2& pos, ImVec2& size){
           ImGuiWindow* window = ImGui::FindWindowByName(last_window);
        if (window == nullptr) return false;
        pos.x = window->Pos.x; pos.y = window->Pos.y + window->Size.y;
        size.x = window->Size.x; size.y = window->Size.y;
        return true;
    }
}

#define wDisplay "Display"
#define wResult "Result"
#define wCells  "Cells"
#define wContractions  "Contractions"
#define wNavigator   "Navigator"
#define wShape "Shape"

using contraction_t = contractionLocator::contraction_t;

                    /************************
                     *
                     *  Setup & Load File
                     *
                     ************************/



    /////////////  lifContext Implementation  ////////////////
// @ todo setup default repository
// default median cover pct is 5% (0.05)

lifContext::lifContext(ci::app::WindowRef& ww, const lif_serie_data& sd, const bfs::path& cache_path, const std::string& fname) :sequencedImageContext(ww), m_serie(sd), m_voxel_view_available(false), mUserStorageDirPath (cache_path), mContentFileName(fname) {
    m_type = guiContext::Type::lif_file_viewer;
    m_show_contractions = false;
    // Create an invisible folder for storage
    mCurrentSerieCachePath = mUserStorageDirPath / m_serie.name();
    bool folder_exists = bfs::exists(mCurrentSerieCachePath);
    std::string msg = folder_exists ? " exists " : " does not exist ";
    msg = " folder for " + m_serie.name() + msg;
    vlogger::instance().console()->info(msg);
    m_idlab_defaults.median_level_set_cutoff_fraction = 15;
    m_playback_speed = 1;
    m_input_selector = input_channel_selector_t(-1,0);
    m_selector_last = -1;
    m_selected_cell = -1;
    m_selected_contraction = -1;
    m_show_display_and_controls = false;
    m_show_results = false;
    m_show_playback = false;
    m_is_loading = false;
    m_content_loaded = false;
    
    m_valid = false;
        m_valid = sd.index() >= 0;
        if (m_valid){
            m_layout = std::make_shared<imageDisplayMapper>  ();
            if (auto lifRef = m_serie.readerWeakRef().lock()){
                m_cur_lif_serie_ref = std::shared_ptr<lifIO::LifSerie>(&lifRef->getSerie(sd.index()), stl_utils::null_deleter());
                setup();
                ww->getRenderer()->makeCurrentContext(true);
            }
    }
}


ci::app::WindowRef&  lifContext::get_windowRef(){
    return shared_from_above()->mWindow;
}

void lifContext::setup_signals(){

    // Create a Processing Object to attach signals to
    m_lifProcRef = std::make_shared<ssmt_processor> (mCurrentSerieCachePath);
    
    // Support lifProcessor::content_loaded
    std::function<void (int64_t&)> content_loaded_cb = boost::bind (&lifContext::signal_content_loaded, shared_from_above(), boost::placeholders::_1);
    boost::signals2::connection ml_connection = m_lifProcRef->registerCallback(content_loaded_cb);
    
    // Support lifProcessor::flu results available
    std::function<void ()> intensity_over_time_ready_cb = boost::bind (&lifContext::signal_intensity_over_time_ready, shared_from_above());
    boost::signals2::connection flu_connection = m_lifProcRef->registerCallback(intensity_over_time_ready_cb);
    
    // Support lifProcessor::initial ss results available
    std::function<void (std::vector<float> &, const input_channel_selector_t&)> root_pci_ready_cb = boost::bind (&lifContext::signal_root_pci_ready, shared_from_above(), boost::placeholders::_1, boost::placeholders::_2);
    boost::signals2::connection nl_connection = m_lifProcRef->registerCallback(root_pci_ready_cb);
    
    // Support lifProcessor::median level set ss results available
    std::function<void (const input_channel_selector_t&)> root_pci_med_reg_ready_cb = boost::bind (&lifContext::signal_root_pci_med_reg_ready, shared_from_above(), boost::placeholders::_1);
    boost::signals2::connection ol_connection = m_lifProcRef->registerCallback(root_pci_med_reg_ready_cb);
    
    // Support lifProcessor::contraction results available
    std::function<void (contractionLocator::contractionContainer_t&,const input_channel_selector_t&)> contraction_ready_cb =
        boost::bind (&lifContext::signal_contraction_ready, shared_from_above(), boost::placeholders::_1, boost::placeholders::_2);
    boost::signals2::connection contraction_connection = m_lifProcRef->registerCallback(contraction_ready_cb);
    
    // Support lifProcessor::geometry_ready
    std::function<void (int,const input_channel_selector_t&)> geometry_ready_cb = boost::bind (&lifContext::signal_regions_ready, shared_from_above(), boost::placeholders::_1, boost::placeholders::_2);
    boost::signals2::connection geometry_connection = m_lifProcRef->registerCallback(geometry_ready_cb);
    
    // Support lifProcessor::temporal_image_ready
    std::function<void (cv::Mat&,cv::Mat&)> ss_segmented_view_ready_cb = boost::bind (&lifContext::signal_segmented_view_ready, shared_from_above(), boost::placeholders::_1, boost::placeholders::_2);
    boost::signals2::connection ss_image_connection = m_lifProcRef->registerCallback(ss_segmented_view_ready_cb);
    
}

void lifContext::setup_params () {
    
//    gl::enableVerticalSync();
 
    
}


void lifContext::setup()
{
    ci::app::WindowRef ww = get_windowRef();
//    ui::Initialize(ui::Options()
//                //   .itemSpacing(vec2(6, 6)) //Spacing between widgets/lines
//                //   .itemInnerSpacing(vec2(10, 4)) //Spacing between elements of a composed widget
//                //   .color(ImGuiCol_Button, ImVec4(0.86f, 0.93f, 0.89f, 0.39f)) //Darken the close button
//               //    .color(ImGuiCol_Border, ImVec4(0.86f, 0.93f, 0.89f, 0.39f))
//                   .window(ww)
//                   );

    m_showGUI = true;
    m_showLog = true;
    m_showHelp = true;
    
    srand( 133 );
    setup_signals();
    assert(is_valid());
    ww->setTitle( m_serie.name());

    mFont = Font( "Menlo", 18 );
    auto ws = ww->getSize();
    mSize = vec2( ws[0], ws[1] / 12);
    m_contraction_names = m_contraction_none;
    clear_playback_params();
    loadCurrentSerie();
    shared_from_above()->update();
    
    ww->getSignalMouseDrag().connect( [this] ( MouseEvent &event ) { processDrag( event.getPos() ); } );
}



/************************
 *
 *
 *  Call Backs
 *
 ************************/

void lifContext::signal_root_pci_ready (std::vector<float> & signal, const input_channel_selector_t& dummy)
{
    stringstream ss;
    ss << svl::toString(dummy.region()) << " root self-similarity available ";
    vlogger::instance().console()->info(ss.str());
}

void lifContext::signal_root_pci_med_reg_ready (const input_channel_selector_t& dummy2)
{
   //@note this is also checked in update. Not sure if this is necessary
    auto tracksRef = m_root_pci_trackWeakRef.lock();
    if ( tracksRef && !tracksRef->at(0).second.empty()){
        m_main_seq.m_time_data.load(tracksRef->at(0), named_colors["PCI"], 2);
    }
    stringstream ss;
    ss << svl::toString(dummy2.region()) << " median regularized root self-similarity available ";
     vlogger::instance().console()->info(ss.str());
}

void lifContext::signal_content_loaded (int64_t& loaded_frame_count )
{
    std::string msg = to_string(mMediaInfo.count) + " Samples in Media  " + to_string(loaded_frame_count) + " Loaded";
    vlogger::instance().console()->info(msg);
    m_is_loading = false;
    m_content_loaded.store(true, std::memory_order_release);
}
void lifContext::signal_intensity_over_time_ready ()
{
    vlogger::instance().console()->info(" Flu Stats Available ");
}

void lifContext::signal_contraction_ready (contractionLocator::contractionContainer_t& contras, const input_channel_selector_t& in)
{
    // Pauses playback if playing and restore it at scope's end
    scopedPause sp(std::shared_ptr<lifContext>(shared_from_above().get(), stl_utils::null_deleter () ));
    
    if (contras.empty()) return;
    
    stringstream ss;
    ss << " Moving Region: " << svl::toString(in.region()) << " " << svl::toString(contras.size());
    vlogger::instance().console()->info(ss.str());
    m_cell2contractions_map[contras[0].m_uid] = contras;
    
// redesign of graphic output
#if NOTYET
    // @note: We are handling one contraction for now.
    m_contractions = contras;
    reset_entire_clip(mMediaInfo.count);
    uint32_t index = 0;
    string name = " C " + svl::toString(index) + " ";
    m_contraction_names.push_back(name);
    auto ctr = m_contractions[0];
    m_clips.emplace_back(ctr.contraction_start.first,
                         ctr.relaxation_end.first,
                         ctr.contraction_peak.first);

    update_sequencer();
#endif
    
}

/*
 * Is called when segmentation is done. The input_selector should have full view (-1) and channel #
 * count is numner
*/
void lifContext::signal_regions_ready(int count, const input_channel_selector_t& in)
{
    assert(m_lifProcRef);
    std::string mr = " Regions Ready: Moving Region";
    mr += (count > 1) ? "s" : " ";
    std::string msg = " Found " + stl_utils::tostr(count) + mr;
    vlogger::instance().console()->info(msg);
    
    m_input_selector = in;
    for (auto mb : m_lifProcRef->moving_bodies()){
        auto roi = mb->roi();
        vlogger::instance().console()->info(" @ " + svl::toString(roi.tl())+"::"+ svl::toString(roi.size()));
        mb->process();
    }
    //@todo parameterize affine windows generation
    // m_lifProcRef->generate_affine_windows();
  
}

void lifContext::glscreen_normalize (const sides_length_t& src, const Rectf& gdr, sides_length_t& dst){
    
    dst.first.x = (src.first.x*gdr.getWidth()) / mMediaInfo.getWidth();
    dst.second.x = (src.second.x*gdr.getWidth()) / mMediaInfo.getWidth();
    dst.first.y = (src.first.y*gdr.getHeight()) / mMediaInfo.getHeight();
    dst.second.y = (src.second.y*gdr.getHeight()) / mMediaInfo.getHeight();
}
                                     
void lifContext::signal_segmented_view_ready (cv::Mat& image, cv::Mat& label)
{
    vlogger::instance().console()->info(" Voxel View Available  ");
    if (! m_lifProcRef){
        vlogger::instance().console()->error("Lif Processor Object does not exist ");
        return;
    }
    m_segmented_image = image.clone();
    m_voxel_view_available = true;
    
    if (! m_segmented_texture ){
        m_segmented_surface = Surface8u::create(cinder::fromOcv(m_segmented_image));
    //    auto cell_ends =  m_lifProcRef->cell_ends ();
     //   m_cell_ends = cell_ends;
    }
}



void lifContext::signal_frame_loaded (int& findex, double& timestamp)
{
    //    frame_indices.push_back (findex);
    //    frame_times.push_back (timestamp);
    //     std::cout << frame_indices.size() << std::endl;
}

void lifContext::fraction_reporter(float f){
    m_fraction_done = f;
    std::cout << (int)(f * 100) << std::endl;
}
/************************
 *
 *  CLIP Processing
 *
 ************************/

int lifContext::get_current_clip_index () const
{
    std::lock_guard<std::mutex> guard(m_clip_mutex);
    return m_current_clip_index;
}

void lifContext::set_current_clip_index (int cindex) const
{
    std::lock_guard<std::mutex> guard(m_clip_mutex);
    m_current_clip_index = cindex;
}

const clip& lifContext::get_current_clip () const {
    return m_clips.at(m_current_clip_index);
}

void lifContext::reset_entire_clip (const size_t& frame_count) const
{
    // Stop Playback ?
    m_clips.clear();
    m_clips.emplace_back(frame_count);
    m_contraction_names.resize(1);
    m_contraction_names[0] = " Entire ";
    set_current_clip_index(0);
}


/************************
 *
 *  Validation, Clear & Log
 *
 ************************/

void lifContext::clear_playback_params ()
{
    m_seek_position = 0;
    m_is_playing = false;
    m_is_looping = false;
    m_show_probe = false;
    m_is_editing = false;
    m_zoom.x = m_zoom.y = 1.0f;
}

bool lifContext::have_lif_serie ()
{
    bool have =   m_serie.readerWeakRef().lock() && m_cur_lif_serie_ref  && mFrameSet ;
    return have;
}


bool lifContext::is_valid () const { return m_valid && is_context_type(guiContext::lif_file_viewer); }




/************************
 *
 *  UI Conrol Functions
 *
 ************************/

void lifContext::loop_no_loop_button ()
{
    if (! have_lif_serie()) return;
    if (looping())
        looping(false);
    else
        looping(true);
}

void lifContext::looping (bool what)
{
    m_is_looping = what;
}


bool lifContext::looping ()
{
    return m_is_looping;
}

void lifContext::play ()
{
    if (! have_lif_serie() || m_is_playing ) return;
    m_is_playing = true;
}

void lifContext::pause ()
{
    if (! have_lif_serie() || ! m_is_playing ) return;
    m_is_playing = false;
}

// For use with RAII scoped pause pattern
void lifContext::play_pause_button ()
{
    if (! have_lif_serie () ) return;
    if (m_is_playing)
        pause ();
    else
        play ();
}



void lifContext::update_sequencer()
{
    m_show_contractions = true;
    m_main_seq.items.resize(1);
    auto ctr = m_contractions[0];

    m_main_seq.items.push_back(timeLineSequence::timeline_item{ 0,
        (int) ctr.contraction_start.first,
        (int) ctr.relaxation_end.first , true});


}

/************************
 *
 *  Seek Processing
 *
 ************************/

// @ todo: indicate mode differently
void lifContext::seekToEnd ()
{
    seekToFrame (get_current_clip().last());
    assert(! m_clips.empty());
}

void lifContext::seekToStart ()
{
    
    seekToFrame(get_current_clip().first());
    assert(! m_clips.empty());
}

int lifContext::getNumFrames ()
{
    return mMediaInfo.count;
}

int lifContext::getCurrentFrame ()
{
    return int(m_seek_position);
}

time_spec_t lifContext::getCurrentTime ()
{
    if (m_seek_position >= 0 && m_seek_position < m_serie.timeSpecs().size())
        return m_serie.timeSpecs()[m_seek_position];
    else return -1.0;
}

time_spec_t lifContext::getStartTime (){
  return m_serie.timeSpecs()[0];
}


void lifContext::seekToFrame (int mark)
{
    const clip& curr = get_current_clip();

    if (mark < curr.first()) mark = curr.first();
    if (mark > curr.last())
        mark = looping() ? curr.first() : curr.last();
    
    m_seek_position = mark;
}


void lifContext::onMarked ( marker_info& t)
{
    seekToFrame((int) t.current_frame());
}

vec2 lifContext::getZoom ()
{
    return m_zoom;
}

void lifContext::setZoom (vec2 zoom)
{
    m_zoom = zoom;
    update ();
}


/************************
 *
 *  Navigation UI
 *
 ************************/

void lifContext::processDrag( ivec2 pos )
{
//    for (Widget* wPtr : mWidgets)
//    {
//        if( wPtr->hitTest( pos ) ) {
//            mTimeMarker.from_norm(wPtr->valueScaled());
//            seekToFrame((int) mTimeMarker.current_frame());
//        }
//    }
}

void  lifContext::mouseWheel( MouseEvent event )
{
#if 0
    if( mMouseInTimeLine )
        mTimeMarker.from_norm(mTimeLineSlider.mValueScaled);
    seekToFrame(mTimeMarker.current_frame());
}

mPov.adjustDist( event.getWheelIncrement() * -5.0f );
#endif

}


void lifContext::mouseMove( MouseEvent event )
{
    update_with_mouse_position (event);
}

void lifContext::update_with_mouse_position ( MouseEvent event )
{
    mMouseInImage = false;
    mMouseInGraphs  = -1;
    
    if (! have_lif_serie () ) return;
    
    mMouseInImage = get_image_display_rect().contains(event.getPos());
    if (mMouseInImage)
    {
        mMouseInImagePosition = event.getPos();
        update_instant_image_mouse ();
    }
    

    
  
}


void lifContext::mouseDrag( MouseEvent event )
{
  
}


void lifContext::mouseDown( MouseEvent event )
{
   
}


void lifContext::mouseUp( MouseEvent event )
{

}




void lifContext::keyDown( KeyEvent event )
{
     ci::app::WindowRef ww = get_windowRef();
    
    if( event.getChar() == 'f' ) {
        setFullScreen( ! isFullScreen() );
    }
    else if( event.getChar() == 'b' ) {
        ww->setBorderless( ! ww->isBorderless() );
    }
    else if( event.getChar() == 't' ) {
        ww->setAlwaysOnTop( ! ww->isAlwaysOnTop() );
    }
    
    
    // these keys only make sense if there is an active movie
    if( have_lif_serie () ) {
        if( event.getCode() == KeyEvent::KEY_LEFT ) {
            pause();
            seekToFrame (getCurrentFrame() - m_playback_speed);
            if (mMouseInImage)
                update_instant_image_mouse ();
        }
        else if( event.getCode() == KeyEvent::KEY_RIGHT ) {
            pause ();
            seekToFrame (getCurrentFrame() + m_playback_speed);
            if (mMouseInImage)
                update_instant_image_mouse ();
        }
        else if( event.getChar() == ' ' ) {
            m_is_playing = true;
        }
        
    }
}

/************************
 *
 *  MedianCutOff Set/Get
 *
 ************************/

void lifContext::setMedianCutOff (int32_t newco)
{
    if (! m_lifProcRef) return;
    // Get a shared_ptr from weak and check if it had not expired
    auto spt = m_lifProcRef->entireContractionWeakRef().lock();
    if (! spt ) return;
    uint32_t tmp = newco % 100; // pct
    uint32_t current (spt->get_median_levelset_pct () * 100);
    if (tmp == current) return;
    spt->set_median_levelset_pct (tmp / 100.0f);
    m_lifProcRef->update(m_input_selector);

}

int32_t lifContext::getMedianCutOff () const
{
    // Get a shared_ptr from weak and check if it had not expired
    auto spt = m_lifProcRef->entireContractionWeakRef().lock();
    if (spt)
    {
        uint32_t current (spt->get_median_levelset_pct () * 100);
        return current;
    }
    return 0;
}


/************************
 *
 *  Load Serie & Setup Widgets
 *
 ************************/

// Set window size according to layout
//  Channel             Data
//  1
//  2
//  3

void lifContext::loadCurrentSerie ()
{
    if ( ! is_valid() || ! m_cur_lif_serie_ref)
        return;
    
    try {
        /*
         * Clear and pause if we are playing back
         */
        m_clips.clear();
        m_instant_channel_display_rects.clear();
        pause();
        
        /*
         * Create the frameset and assign the channel names
         * Fetch the media info
         */
        mFrameSet = seqFrameContainer::create (*m_cur_lif_serie_ref);
        if (! mFrameSet || ! mFrameSet->isValid())
        {
            vlogger::instance().console()->debug("Serie had 1 or no frames ");
            return;
        }
        mMediaInfo = mFrameSet->media_info();
        mChannelCount = (uint32_t) mMediaInfo.getNumChannels();
        m_instant_channel_display_rects.resize(mChannelCount);
        
        if (!(mChannelCount > 0 && mChannelCount < 4)){
            vlogger::instance().console()->debug("Expected 1 or 2 or 3 channels ");
            return;
        }

        m_layout->init ( mFrameSet->media_info());

        /*
         * Create the frameset and assign the channel namesStart Loading Images on async on a different thread
         * Loading also produces voxel images.
         */
        m_plot_names.clear();
        m_plot_names = m_serie.channel_names();
        m_plot_names.push_back("MisRegister");

        m_content_loaded.store(false, std::memory_order_release);
        auto load_thread = std::thread(&ssmt_processor::load, m_lifProcRef.get(),mFrameSet, m_serie);
        load_thread.detach();
        m_is_loading = true;

        /*
         * Fetch length and channel names
         */
        mFrameSet->channel_names (m_serie.channel_names());
        reset_entire_clip(mFrameSet->count());
        m_minFrame = 0;
        m_maxFrame =  mFrameSet->count() - 1;
        
        // Initialize The Main Sequencer
        m_main_seq.mFrameMin = 0;
        m_main_seq.mFrameMax = (int) mFrameSet->count() - 1;
        m_main_seq.mSequencerItemTypeNames = {"RGG"};
        m_main_seq.items.push_back(timeLineSequence::timeline_item{ 0, 0, (int) mFrameSet->count(), true});
        
        m_title = m_serie.name() + " @ " + mContentFileName;
        
        auto ww = get_windowRef();
        ww->setTitle(m_title );
        ww->getApp()->setFrameRate(mMediaInfo.getFramerate());
        
        
        mFrameSize = mMediaInfo.getSize();
        
        mSurface = Surface8u::create (int32_t(mFrameSize.x), int32_t(mFrameSize.y), true);
        mFbo = gl::Fbo::create(mSurface->getWidth(), mSurface->getHeight());
        m_show_playback = true;
    }
    catch( const std::exception &ex ) {
        console() << ex.what() << endl;
        m_show_playback = false;
        return;
    }
}

void lifContext::process_async (){
    
    while(!m_content_loaded.load(std::memory_order_acquire)){
           std::this_thread::yield();
       }
    
     auto load_thread = std::thread(&ssmt_processor::find_moving_regions, m_lifProcRef.get(),channel_count()-1);
       load_thread.detach();

    
    progress_fn_t pf = std::bind(&lifContext::fraction_reporter, this, std::placeholders::_1);
    switch(channel_count()){
        case 3:
        {
            // note launch mode is std::launch::async
            m_intensity_tracks_aync = std::async(std::launch::async,&ssmt_processor::run_intensity_statistics,
                                                  m_lifProcRef.get(), std::vector<int> ({0,1}) );
            input_channel_selector_t in (-1,2);
            m_root_pci_tracks_asyn = std::async(std::launch::async, &ssmt_processor::run_selfsimilarity_on_selected_input, m_lifProcRef.get(), in, pf);
            break;
        }
        case 2:
        {
             //note launch mode is std::launch::async
            m_intensity_tracks_aync = std::async(std::launch::async,&ssmt_processor::run_intensity_statistics,
                                                    m_lifProcRef.get(), std::vector<int> ({0}) );
            input_channel_selector_t in (-1,1);
            m_root_pci_tracks_asyn = std::async(std::launch::async, &ssmt_processor::run_selfsimilarity_on_selected_input, m_lifProcRef.get(), in, pf);
            break;
        }
        case 1:
        {
            input_channel_selector_t in (-1,0);
            m_root_pci_tracks_asyn = std::async(std::launch::async, &ssmt_processor::run_selfsimilarity_on_selected_input, m_lifProcRef.get(), in, pf);
            break;
        }
    }
}


/************************
 *
 *  Update & Draw
 *
 ************************/


const Rectf& lifContext::get_image_display_rect ()
{
    return m_display_rect;
}

void  lifContext::update_channel_display_rects (){
    int cn = channel_count();
    assert(m_instant_channel_display_rects.size() == channel_count());
    vec2 channel_size (get_image_display_rect().getWidth(), get_image_display_rect().getHeight()/cn);
    vec2 itl = get_image_display_rect().getUpperLeft();
    for (int cc = 1; cc <= cn; cc++){
        Rectf& rect = m_instant_channel_display_rects[cc-1];
        rect = Rectf(itl, itl+channel_size);
        itl += ivec2(0, channel_size.y);
    }
}

const Rectf& lifContext::get_channel_display_rect (const int channel_number_zero_based){
    return m_instant_channel_display_rects[channel_number_zero_based];
}


void lifContext::add_canvas (){
   
    //@note: In ImGui, AddImage is called with uv parameters to specify a vertical flip.
    auto showImage = [&](const char *windowName,bool *open, const gl::Texture2dRef texture){
        if (open && *open)
        {
          //  ImGuiIO& io = ImGui::GetIO();
           // ImGui::SetNextWindowBgAlpha(0.4f); // Transparent background
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            if (ImGui::Begin(windowName, open)) //, io.ConfigResizeWindowsFromEdges))
            {
                ImVec2 pos = ImGui::GetCursorScreenPos(); // actual position
                draw_list->AddImage(  reinterpret_cast<ImTextureID> (texture->getId()), pos,
                                    ImVec2(ImGui::GetContentRegionAvail().x + pos.x, ImGui::GetContentRegionAvail().y  + pos.y),
                                    ImVec2(0,1), ImVec2(1,0));
                ImVec2 canvas_pos = ImGui::GetCursorScreenPos();            // ImDrawList API uses screen coordinates!
                ImVec2 canvas_size = ImGui::GetContentRegionAvail();        // Resize canvas to what's available
                ImRect regionRect(canvas_pos, canvas_pos + canvas_size);
                // Update image/display coordinate transformer
                m_layout->update_display_rect(canvas_pos, canvas_pos + canvas_size);
                // Draw Divider between channels
                ImVec2 midv = m_layout->image2display(ivec2(0,mMediaInfo.channel_size.height));
                ImVec2 midh = m_layout->image2display(ivec2(mMediaInfo.channel_size.width, mMediaInfo.channel_size.height));
                draw_list->AddLine(midv, midh, IM_COL32(0,0,128,128), 3.0f);
                
                int index = 0;
                auto d_channel = channel_count() - 1;
                for (const auto& mb : m_lifProcRef->moving_bodies()){
                    const Point2f& pc = mb->motion_surface().center;
                    ivec2 iv(pc.x, pc.y);
                    
                    ImVec2 ic = m_layout->image2display(iv, d_channel, true);
                    auto roi = mb->roi();
                    auto tl = m_layout->image2display(ivec2(roi.x, roi.y),  d_channel);
                    DrawCross(draw_list, ImVec4(0.0f, 1.0f, 0.0f, 0.5f), ImVec2(16,16), false, tl);
                    auto selected = DrawCross(draw_list, ImVec4(1.0f, 0.0f, 0.0f, 0.5f), ImVec2(16,16),
                                              m_selected_cell == index, ic);
                    if(selected == 2){
                        if (m_selected_cell < 0)m_selected_cell = index;
                        else m_selected_cell = -1;
                        auto sel_string = m_selected_cell < 0 ? " is deselected " : " is selected";
                        auto msg = " Cell " + tostr(index) + sel_string;
                        vlogger::instance().console()->info(msg);
                    }

                    if(! mb->poly().empty()){
                        std::vector<ImVec2> impts;
                        std::transform(mb->poly().begin(), mb->poly().end(), back_inserter(impts), [&](const cv::Point& f)->ImVec2
                                       { ivec2 v(f.x,f.y); return m_layout->image2display(v, d_channel); });
                        auto nc = numbered_half_colors[index];
                        draw_list->AddConvexPolyFilled( impts.data(), impts.size(), nc);
                    }
                    index++;
                }
            }
            ImGui::End();
        }
    };
    
    //@note: assumes, next image plus any on image graphics have already been added
    // offscreen via FBO
    if(m_show_playback){
        ImVec2 size;
        ImVec2 pos (300,18);
        getPosSizeFromWindow(mContentFileName.c_str(), pos, size);
        size.x = mFrameSize.x+mFrameSize.x/2;
        size.y = mFrameSize.y+mFrameSize.y/2;
        auto ww = get_windowRef();

        ScopedWindowWithFlag utilities(wDisplay, nullptr, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_HorizontalScrollbar );
        ImGui::SetNextWindowPos(pos);
        ImGui::SetNextWindowSize(size);
        ImGui::SameLine();
        showImage(wDisplay, &m_show_playback, mFbo->getColorTexture());
    }
}

/*
 * Result Window, to below the Display
 * Size width: remainder to the edge of app window in x ( minus a pad )
 * Size height: app window height / 2
 */

void lifContext::add_result_sequencer (){
 
    // let's create the sequencer
    static int selectedEntry = -1;
    static int64 firstFrame = 0;
    static bool expanded = true;
    ImVec2 pos, size;
    assert(getPosSizeFromWindow(wDisplay, pos, size));
    m_results_browser_display = Rectf(glm::vec2(pos.x,pos.y),glm::vec2(size.x,size.y));
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);

    static bool results_open;
    if(ImGui::Begin(wResult, &results_open, ImGuiWindowFlags_AlwaysAutoResize)){
        Sequencer(&m_main_seq, &m_seek_position, &expanded, &selectedEntry, &firstFrame, ImSequencer::SEQUENCER_EDIT_NONE );
    }
    ImGui::End();
//    // add a UI to edit that particular item
//    if (selectedEntry != -1)
//    {
//        const timeLineSequence::timeline_item &item = m_main_seq.items[selectedEntry];
//        ImGui::Text("I am a %s, please edit me", m_main_seq.mSequencerItemTypeNames[item.mType].c_str());
//        // switch (type) ....
//    }
}


void lifContext::add_navigation(){
    
    if(m_show_playback){
        
        ImGuiWindow* window = ImGui::FindWindowByName(wResult);
        assert(window != nullptr);
        ImVec2 pos (window->Pos.x, window->Pos.y + window->Size.y );
        ImVec2 size (window->Size.x, 100);
        
        ScopedWindowWithFlag utilities(wNavigator, nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        m_navigator_display = Rectf(glm::vec2(pos.x,pos.y),glm::vec2(size.x,size.y));
        ImGui::SetNextWindowPos(pos);
        ImGui::SetNextWindowSize(size);
        ImGui::SameLine();
        ImGui::BeginGroup();
        {
            ImGui::PushItemWidth(40);
            ImGui::PushID(200);
            ImGui::InputInt("", &m_main_seq.mFrameMin, 0, 0);
            ImGui::PopID();
            ImGui::SameLine();
            if (ImGui::Button("|<"))
                seekToStart();
            ImGui::SameLine();
            if (ImGui::Button("<"))
                seekToFrame (getCurrentFrame() - m_playback_speed);
            ImGui::SameLine();
            if (ImGui::Button(">"))
                seekToFrame (getCurrentFrame() + m_playback_speed);
            ImGui::SameLine();
            if (ImGui::Button(">|"))
                seekToEnd();
            ImGui::SameLine();
            if (ImGui::Button(m_is_playing ? "Stop" : "Play"))
            {
                play_pause_button();
            }
            
            if(mNoLoop == nullptr){
                mNoLoop = gl::Texture::create( loadImage( loadResource( IMAGE_PNLOOP  )));
            }
            if (mLoop == nullptr){
                mLoop = gl::Texture::create( loadImage( loadResource( IMAGE_PLOOP )));
            }
            
            unsigned int playNoLoopTextureId = mNoLoop->getId();//  evaluation.GetTexture("Stock/PlayNoLoop.png");
            unsigned int playLoopTextureId = mLoop->getId(); //evaluation.GetTexture("Stock/PlayLoop.png");
            
            ImGui::SameLine();
            if (ImGui::ImageButton((ImTextureID)(uint64_t)(m_is_looping ? playLoopTextureId : playNoLoopTextureId), ImVec2(16.f, 16.f)))
                loop_no_loop_button();
            
            ImGui::SameLine();
            ImGui::PushID(202);
            ImGui::InputInt("",  &m_main_seq.mFrameMax, 0, 0);
            ImGui::PopID();
            ImGui::SameLine();
            if (ImGui::Button(m_playback_speed == 10 ? " 1 " : " 10x "))
            {
                auto current = m_playback_speed;
                m_playback_speed = current == 1 ? 10 : 1;
            }
            ImGui::SameLine();
            
            auto dt = getCurrentTime().secs() - getStartTime().secs();
            ui::SameLine(0,0); ui::Text("% 8d\t%4.4f Seconds", getCurrentFrame(), dt);
            
            if(!m_root_pci_trackWeakRef.expired()){
                if(m_median_set_at_default){
                    int default_median_cover_pct = m_idlab_defaults.median_level_set_cutoff_fraction;
                    setMedianCutOff(default_median_cover_pct);
                    ImGui::SliderInt("Median Cover Percent", &default_median_cover_pct, default_median_cover_pct, default_median_cover_pct);
                }
                else{
                    int default_median_cover_pct = getMedianCutOff();
                    ImGui::SliderInt("Median Cover Percent", &default_median_cover_pct, 0, 20);
                    setMedianCutOff(default_median_cover_pct);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button(m_median_set_at_default ? "Default" : "Not Set"))
            {
                m_median_set_at_default = ! m_median_set_at_default;
            }
            
        }
        ImGui::EndGroup();
    }
}


/*
 * Segmentation image is reduced from full resolution. It is exanpanded by displaying it in full res here
 */
void lifContext::add_motion_profile (){
    
 if (! m_voxel_view_available ) return;
    if (! m_segmented_texture && m_segmented_surface){
        // Create a texcture for display
        auto texFormat = gl::Texture2d::Format().loadTopDown();
        m_segmented_texture = gl::Texture::create(*m_segmented_surface, texFormat);
    }
    
    ImGuiWindow* window = ImGui::FindWindowByName(wDisplay);
    assert(window != nullptr);
    auto ww = get_windowRef();
    ImVec2  sz (m_segmented_texture->getWidth(),m_segmented_texture->getHeight());
    ImVec2  frame (mMediaInfo.channel_size.width, mMediaInfo.channel_size.height);
    ImVec2 pos (ww->getSize().x - sz.x, ww->getSize().y - sz.y);
    m_motion_profile_display = Rectf(glm::vec2(pos.x,pos.y),glm::vec2(frame.x,frame.y));
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowContentSize(frame);
    
    if (ImGui::Begin(wShape, nullptr ))
    {
        if(m_segmented_texture){
            ImGui::BeginChild(" ", frame, true );
            ImGui::Image( (void*)(intptr_t) m_segmented_texture->getId(), frame);
            ImGui::EndChild();
        }
    }
    ImGui::End();
}




// Right Hand Size:
// Cells
// Contractions
// Shape

void lifContext::draw_contraction_plots(const contractionLocator::contraction_t& ct, int id){
    contraction_t::sigContainer_t force = ct.force;
    auto elon = ct.elongation;
    auto elen = ct.interpolated_length;
    svl::norm_min_max (force.begin(), force.end(), true);
    svl::norm_min_max (elon.begin(), elon.end(), true);
    svl::norm_min_max (elen.begin(), elen.end(), true);
    
//    static const float* y_data[] = force.data();
    static ImU32 colors[3] = { ImColor(0, 255, 0), ImColor(255, 0, 0), ImColor(0, 0, 255) };
    std::vector<float> x_data;
    std::generate(x_data.begin(), x_data.end(), [] {
        static int i = 0;
        return i++;
    });
    std::string names [] = { " Force ", " Elongation ", " Interpolated Length " };

    auto plot = [=](const contraction_t::sigContainer_t& values, std::string& name, ImU32 color,int framewidth, int frameheight){
        static uint32_t selection_start = 0, selection_length = 0;

        ImGui::PlotConfig conf;
        conf.values.xs = nullptr; //x_data.data();
        conf.values.ys = values.data();
        conf.values.count = values.size();
        conf.values.color = colors[0];
        conf.scale.min = 0;
        conf.scale.max = 1;
        conf.tooltip.show = true;
        conf.selection.show = true;
        conf.selection.start = &selection_start;
        conf.selection.length = &selection_length;
        conf.frame_size = ImVec2(framewidth, frameheight);
        conf.skip_small_lines = false;
        conf.overlay_text = name.c_str();
        ImGui::Plot(name.c_str(), conf);
        
    };
    std::string title = " Cell/Tissue " + stl_utils::tostr(id) + " ";
    ImGui::Begin(title.c_str(), nullptr, 0); //ImGuiWindowFlags_AlwaysAutoResize);
    plot(force, names[0], colors[0], 128, 200);
    ImGui::SameLine();
    plot(elon, names[1], colors[1], 128,200);
    ImGui::SameLine();
    plot(elen, names[2], colors[2], 128, 200);
    ImGui::End();
    

}
void lifContext::add_contractions (bool* p_open)
{
    if (m_lifProcRef->moving_regions().empty()) return;
    
    ImGuiWindow* window = ImGui::FindWindowByName(wDisplay);
    assert(window != nullptr);
    ImVec2 pos (window->Pos.x + window->Size.x, window->Pos.y );
    ImGui::SetNextWindowPos(pos);
    auto ww = get_windowRef();
    ImGui::SetNextWindowSize(ImVec2(ww->getSize().x/2,ww->getSize().y/4), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(wCells, p_open, ImGuiWindowFlags_MenuBar)){
        for (int i = 0; i < m_lifProcRef->moving_bodies().size(); i++){
            const ssmt_result::ref_t& mb = m_lifProcRef->moving_bodies()[i];
            auto contractions = m_cell2contractions_map[mb->id()];
#ifdef NotYet
            static ImGuibfs::Dialog* dialog = nullptr;
             const bool browseButtonPressed = ImGui::Button(" Export CSV ");
                 if (browseButtonPressed) {
                    if (dialog != nullptr)
                         delete dialog;
                     dialog = new ImGuibfs::Dialog();
                 }
                 if (dialog) {
                     static std::string title = " Cell / Contraction Info ";
                     dialog->chooseFolderDialog(browseButtonPressed,mCurrentSerieCachePath.c_str(), title.c_str());
                     // @note Export CSV
                 }
#endif
            if (ImGui::TreeNode((void*)(intptr_t)i, "Cell/Tissue %d", mb->id())){
                
                for (int cc = 0; cc < contractions.size(); cc++){
                    const auto ct = contractions[cc];
                    ImGui::Text(" Contraction Start : %d", int(ct.contraction_start.first));
                    ImGui::Text(" Contraction Peak : %d", int(ct.contraction_peak.first));
                    ImGui::Text(" Relaxation End : %d", int(ct.relaxation_end.first));
                }
                ImGui::TreePop();
            }
        }
        ImGui::End();
    }
    for (int i = 0; i < m_lifProcRef->moving_bodies().size(); i++){
         const ssmt_result::ref_t& mb = m_lifProcRef->moving_bodies()[i];
         auto contractions = m_cell2contractions_map[mb->id()];
         for (int cc = 0; cc < contractions.size(); cc++){
                        const auto ct = contractions[cc];
             draw_contraction_plots(ct, mb->id());
         }
    }
}



void  lifContext::DrawGUI(){
    
    add_canvas();
    add_result_sequencer();
    add_navigation();
    m_show_cells = true;
    add_contractions(&m_show_cells);
    add_motion_profile ();
}

void  lifContext::update_log (const std::string& msg)
{
    if (msg.length() > 2)
        mLog = msg;
    TextBox tbox = TextBox().alignment( TextBox::RIGHT).font( mFont ).size( mSize ).text( mLog );
    tbox.setColor( Color( 1.0f, 0.65f, 0.35f ) );
    tbox.setBackgroundColor( ColorA( 0.3f, 0.3f, 0.3f, 0.4f )  );
    tbox.measure();
    mTextTexture = gl::Texture2d::create( tbox.render() );
}

//@todo implement
void lifContext::resize ()
{
    if (! have_lif_serie () || ! mSurface ) return;
}

bool lifContext::haveTracks()
{
    return ! m_intensity_trackWeakRef.expired() && ! m_root_pci_trackWeakRef.expired();
}


void lifContext::update ()
{
    
    if (! have_lif_serie() ) return;


    if ( is_ready (m_intensity_tracks_aync) )
        m_intensity_trackWeakRef = m_intensity_tracks_aync.get();
    

    auto flu_cnt = channel_count() - 1;
    // Update Fluorescence results if ready
    if (flu_cnt > 0 && ! m_intensity_trackWeakRef.expired())
    {
        // Number of Flu runs is channel count - 1
        auto tracksRef = m_intensity_trackWeakRef.lock();
        for (auto cc = 0; cc < flu_cnt; cc++){
            m_main_seq.m_time_data.load(tracksRef->at(cc), named_colors[tracksRef->at(cc).first], cc);
        }
    }

    
    // Update PCI result if ready
    if ( is_ready (m_root_pci_tracks_asyn)){
        m_root_pci_trackWeakRef = m_root_pci_tracks_asyn.get();
    }
    
    if ( ! m_root_pci_trackWeakRef.expired())
    {
        auto tracksRef = m_root_pci_trackWeakRef.lock();
        if(tracksRef && tracksRef->size() > 0 && ! tracksRef->at(0).second.empty())
            m_main_seq.m_time_data.load(tracksRef->at(0), named_colors["PCI"], channel_count()-1);
    }
    
    // Fetch Next Frame
    if (have_lif_serie ()){
        mSurface = mFrameSet->getFrame(getCurrentFrame());
        mCurrentIndexTime = mFrameSet->currentIndexTime();
        if (mCurrentIndexTime.first != m_seek_position){
            vlogger::instance().console()->info(tostr(mCurrentIndexTime.first - m_seek_position));
        }
        // Update Fbo with texture.
        {
            lock_guard<mutex> scopedLock(m_update_mutex);
            if(mSurface){
                mFbo->getColorTexture()->update(*mSurface);
                renderToFbo(mSurface, mFbo);
            }
        }
    }
    
    if (m_is_playing )
    {
        update_instant_image_mouse ();
        seekToFrame (getCurrentFrame() + m_playback_speed);
    }
    
    // Update text texture with most recent text
    auto num_str = to_string( int( getCurrentFrame ()));
    std::string frame_str = m_is_playing ? string( "Playing: " + num_str) : string ("Paused: " + num_str);
    update_log (frame_str);
}

//@note:
// LIF 3 channel organization. Channel height is 1/3 of image height
// Channel Index is pos.y / channel_height,
// In channel x is pos.x, In channel y is pos.y % channel_height
void lifContext::update_instant_image_mouse ()
{
    auto image_pos = m_layout->display2image(mMouseInImagePosition);
    uint32_t channel_height = mMediaInfo.getChannelSize().y;
    m_instant_channel = ((int) image_pos.y) / channel_height;
    m_instant_mouse_image_pos = image_pos;
    if (mSurface)
    {
        m_instant_pixel_Color = mSurface->getPixel(m_instant_mouse_image_pos);
    }
    
    
}

gl::TextureRef lifContext::pixelInfoTexture ()
{
    if (! mMouseInImage) return gl::TextureRef ();
    TextLayout lout;
    const auto names = m_serie.channel_names();
    auto channel_name = (m_instant_channel < names.size()) ? names[m_instant_channel] : " ";
    
    // LIF has 3 Channels.
    uint32_t channel_height = mMediaInfo.getChannelSize().y;
    std::string pos = " " + svl::toString(((int)m_instant_pixel_Color.g)) + " @ [ " +
    to_string(imagePos().x) + "," + to_string(imagePos().y % channel_height) + "]";
    vec2                textSize(250, 100);
    TextBox tbox = TextBox().alignment( TextBox::LEFT).font( mFont ).size( ivec2( textSize.x, TextBox::GROW ) ).text( pos );
    tbox.setFont( Font( "Times New Roman", 34 ) );
    if (channel_name == "Red")
        tbox.setColor( ColorA(0.8,0.2,0.1,1.0) );
    else if (channel_name == "Green")
        tbox.setColor( ColorA(0.2,0.8,0.1,1.0) );
    else
        tbox.setColor( ColorA(0.0,0.0,0.0,1.0) );
    tbox.setBackgroundColor( ColorA( 0.3, 0.3, 0.3, 0.3 ) );
    ivec2 sz = tbox.measure(); sz.x = sz.y;
    return  gl::Texture2d::create( tbox.render() );
}


void lifContext::draw_info ()
{
   
    auto ww = get_windowRef();
    auto ws = ww->getSize();
    gl::setMatricesWindow( ws );


    gl::ScopedBlendAlpha blend_;

    
    if (m_show_probe)
    {
        auto texR = pixelInfoTexture ();
        if (texR)
        {
            Rectf tbox = texR->getBounds();
            tbox.offset(mMouseInImagePosition);
            gl::draw( texR, tbox);
        }
    }
    
    if (mTextTexture)
    {
        Rectf textrect (0.0, ws[1] - mTextTexture->getHeight(), ws[0],ws[1]);
        gl::draw(mTextTexture, textrect);
    }
    
}
void  lifContext::renderToFbo (const SurfaceRef&, gl::FboRef& fbo ){
    // this will restore the old framebuffer binding when we leave this function
    // on non-OpenGL ES platforms, you can just call mFbo->unbindFramebuffer() at the end of the function
    // but this will restore the "screen" FBO on OpenGL ES, and does the right thing on both platforms
    gl::ScopedFramebuffer fbScp( fbo );
    
}

void lifContext::draw (){
    if( have_lif_serie()  && mSurface ){
        
        if(m_showGUI)
            DrawGUI();
    }
}
 
    
#pragma GCC diagnostic pop
