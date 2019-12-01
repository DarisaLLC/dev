#ifndef __movContext___h
#define __movContext___h

#include "guiContext.h"
#include "clipManager.hpp"
#include "visible_layout.hpp"
#include <atomic>
#include <memory>
#include "OnImagePlotUtils.h"
#include "imGuiCustom/ImSequencer.h"
#include "imGuiCustom/visibleSequencer.h"
#include "cvVideoPlayer.h"
#include "ssmt.hpp"


using namespace boost;
using namespace boost::filesystem;
using namespace ci;
using namespace ci::app;
using namespace ci::signals;

using namespace params;

using namespace std;
namespace fs = boost::filesystem;

typedef std::shared_ptr<class movContext> movContextRef;



class movContext : public sequencedImageContext
{

public:
    
  
    
    // From a image_sequence_data
    movContext(ci::app::WindowRef& ww, const cvVideoPlayer::ref&, const fs::path&  );
    
    // shared_from_this() through base class
    movContext* shared_from_above () {
        return static_cast<movContext*>(((guiContext*)this)->shared_from_this().get());
    }

    
	static const std::string& caption () { static std::string cp ("Media Viewer # "); return cp; }
	virtual void draw () override;
	virtual void setup () override;
	virtual bool is_valid () const override;
	virtual void update () override;
	virtual void resize () override;
	void draw_window ();
	void draw_info () override;
	
	virtual void onMarked (marker_info_t&) ;
	virtual void mouseDown( MouseEvent event ) override;
	virtual void mouseMove( MouseEvent event ) override;
	virtual void mouseUp( MouseEvent event ) override;
	virtual void mouseDrag( MouseEvent event ) override;
	virtual void mouseWheel( MouseEvent event ) ;
	virtual void keyDown( KeyEvent event ) override;
	
	virtual void seekToStart () override;
	virtual void seekToEnd () override;
	virtual void seekToFrame (int) override;
	virtual int getCurrentFrame () override;
	virtual time_spec_t getCurrentTime () override;
    virtual time_spec_t getStartTime () override;
	virtual int getNumFrames () override;
	virtual void processDrag (ivec2 pos) override;
    
    bool haveTracks();
	
	const ivec2& imagePos () const { return m_instant_mouse_image_pos; }
	const uint32_t& channelIndex () const { return m_instant_channel; }
	
	void setZoom (vec2);
	vec2 getZoom ();
	void setMedianCutOff (int32_t newco);
	int32_t getMedianCutOff () const;
    void setCellLength (uint32_t newco);
    uint32_t getCellLength () const;


	
	void play_pause_button ();
	void loop_no_loop_button ();
    void edit_no_edit_button ();
	
	const tiny_media_info& media () const { return mMediaInfo; }
    const uint32_t& channel_count () const { return mChannelCount; }
    
    // Navigation
    void play ();
    void pause ();
    bool isPlaying () const { return m_is_playing; }
    
    // Supporting gui_base
    void SetupGUIVariables() override;
    void DrawGUI() override;
    void QuitApp();
    
private:
    void renderToFbo (const SurfaceRef&, gl::FboRef& fbo );
    void add_canvas ();
    void setup_signals ();
    void setup_params ();
    ci::app::WindowRef& get_windowRef();
    const Rectf& get_channel_display_rect (const int channel_number);
    void glscreen_normalize (const sides_length_t& src, const Rectf& gdr, sides_length_t& dst);
    
    // mov Support
    std::shared_ptr<ssmt_processor> m_movProcRef;
	void load_current_sequence ();
	bool have_sequence ();
    cvVideoPlayer::ref m_sequence_player_ref;
    boost::filesystem::path mPath;
    std::vector<std::string> m_plot_names;
    
    
    // Callbacks
    void signal_content_loaded (int64_t&);
    void signal_sm1d_ready (std::vector<float> &, const input_channel_selector_t&);
    void signal_sm1dmed_ready (const input_channel_selector_t&);
    void signal_frame_loaded (int& findex, double& timestamp);
    void signal_segmented_view_ready (cv::Mat& image, cv::Mat& label);
    
    // Availability
    std::atomic<bool> m_geometry_available;
    // Availability
      std::atomic<bool> m_voxel_view_available;
      input_channel_selector_t  m_input_selector;
      mutable std::atomic<int> m_selector_last;


    // Clip Processing
    int get_current_clip_index () const;
    void set_current_clip_index (int cindex) const;
    void reset_entire_clip (const size_t& frame_count) const;
    const clip& get_current_clip () const;
    mutable volatile int m_current_clip_index;
    mutable std::vector<clip> m_clips;
    mutable std::mutex m_clip_mutex;
    mutable std::mutex m_update_mutex;
    
    // Async Processing
    void process_async ();
    void process_window(int);
    
     
    // Frame Cache and frame store
    std::shared_ptr<seqFrameContainer> mFrameSet;
    SurfaceRef  mSurface;
  // Variance Image & Texture
    cv::Mat m_segmented_image;
    ci::gl::TextureRef m_segmented_texture;
    SurfaceRef m_segmented_surface;
    
    // Tracks of frame associated results
    std::weak_ptr<vecOfNamedTrack_t> m_basic_trackWeakRef;
    std::weak_ptr<vecOfNamedTrack_t> m_longterm_pci_trackWeakRef;

    // Folder for Per user result / content caching
    // @todo clarify the difference. Right now they are the same
    boost::filesystem::path mUserStorageDirPath;
    boost::filesystem::path mCurrentCachePath;
    
    // Content Info
    tiny_media_info      mMediaInfo;
    mutable uint32_t  mChannelCount;
    int64_t m_minFrame;
    int64_t m_maxFrame;
    
    // Instant information
    int64_t m_seek_position;
    bool m_is_playing, m_is_looping;
    bool m_is_editing, m_show_probe;
    bool m_is_in_params;
    vec2 m_zoom;
    vec2 mMousePos;
    bool mMouseIsDown;
    bool mMouseIsMoving;
    bool mMouseIsDragging;
    bool mMetaDown;
    int mMouseInGraphs; // -1 if not, 0 1 2
    bool mMouseInImage; // if in Image, mMouseInGraph is -1
    ivec2 mMouseInImagePosition;
    uint32_t m_cutoff_pct;	

    gl::TextureRef mImage;

    // Screen Info
    vec2 mScreenSize;
    gl::TextureRef pixelInfoTexture ();
    std::string m_title;
    
    // Update GUI
    void clear_playback_params ();
    void update_instant_image_mouse ();
    void update_with_mouse_position ( MouseEvent event );
    const Rectf& get_image_display_rect () override;
    void update_channel_display_rects () override;
    
    vec2 texture_to_display_zoom ();
    void add_result_sequencer ();
    void add_navigation ();
    void add_motion_profile ();
    
    // Navigation
    void update_log (const std::string& meg = "") override;
    bool looping () override;
    void looping (bool what) override;
	void seek( size_t xPos );

	
    // Layout Manager
    std::shared_ptr<imageDisplayMapper> m_layout;
  
    // UI flags
    bool m_showLog, m_showGUI, m_showHelp;
    bool m_show_contractions, m_show_playback;
    bool m_median_set_at_default;
    
    
    // imGui
    timeLineSequence m_result_seq;
    void draw_sequencer ();
    
    // UI instant sub-window rects
    Rectf m_results_browser_display;
    Rectf m_navigator_display;
      Rectf m_motion_profile_display;
    int8_t m_playback_speed;
    
    OnImagePlot m_tsPlotter;

    // User Selectable ROIs
    std::map<int,Rectf> m_windows;
    
    // Resource Icons
    ci::gl::TextureRef    mNoLoop, mLoop;
    

    
	static size_t Normal2Index (const Rectf& box, const size_t& pos, const size_t& wave)
	{
		size_t xScaled = (pos * wave) / box.getWidth();
		xScaled = svl::math<size_t>::clamp( xScaled, 0, wave );
		return xScaled;
	}
	

    class scopedPause : private Noncopyable {
    public:
        scopedPause (const movContextRef& ref) :weakContext(ref) {
            if (!weakContext){
                if (weakContext->isPlaying())
                    weakContext->play_pause_button();
            }
        }
        ~scopedPause (){
            if (!weakContext ){
                if (!weakContext->isPlaying()){
                    weakContext->play_pause_button();
                }
            }
        }
    private:
        std::shared_ptr<movContext>weakContext;
    };


};

#endif
