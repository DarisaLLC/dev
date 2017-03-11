

#ifndef __VISIBLE_LAYOUT__
#define __VISIBLE_LAYOUT__

#include "cinder/Rect.h"
#include "cinder/Signals.h"
#include "cinder/app/Event.h"

#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace ci;
using namespace ci::signals;

class layout : tiny_media_info
{
public:
    
    typedef	 signals::Signal<void( ivec2 & )>		LayoutSignalWindowSize_t;
    LayoutSignalWindowSize_t&	getSignalWindowSize() { return m_windowSizeSignal; }
    
    layout (ivec2 trim, bool keep_aspect = true):m_keep_aspect ( keep_aspect), m_trim (trim), m_isSet (false)
    {
        
        m_image_frame_size_norm = vec2(0.75, 0.75);
        m_single_plot_size_norm = vec2(0.25, 0.25);
        
 
    }
    
    // Constructor
    void init (const WindowRef& uiWin, const tiny_media_info& tmi)

    {
        *((tiny_media_info*)this) = tmi;
        m_canvas_size = uiWin->getSize ();
        Area wi (0, 0, uiWin->getSize().x, uiWin->getSize().y);
        m_window_rect = Rectf(wi);
        Area ai (0, 0, tmi.getWidth(), tmi.getHeight());
        m_image_rect = Rectf (ai);
        m_aspect = layout::aspect(tmi.getSize());
        m_isSet = true;
      }

    inline bool isSet () const { return m_isSet; }
    
    inline void update_window_size (const ivec2& new_size )
    {
        ivec2 ns = new_size;
        if (m_keep_aspect)
        {
            float newAr = new_size.x / (float) new_size.y ;
            if (aspectRatio() > newAr)
                ns.y= std::floor(ns.x/ aspectRatio ());
            else
                ns.x = std::floor(ns.y * aspectRatio () );
        }
        
        m_canvas_size = new_size;
        m_windowSizeSignal.emit(m_canvas_size);
    }
    
    inline Rectf image_frame_rect ()
    {
        
        vec2 tl = image_frame_position();
        vec2 br = vec2 (tl.x + image_frame_size().x, tl.y + image_frame_size().y);
        Rectf fr (tl, br);
        auto ir = m_image_rect.getCenteredFit(fr, true);
        return ir;
    }
    
    // Modify window size
    inline void scale (vec2& scale_by)
    {
        m_canvas_size.x  = std::floor(m_canvas_size.x * scale_by.x);
        m_canvas_size.y  = std::floor(m_canvas_size.y * scale_by.y);
        m_windowSizeSignal.emit(m_canvas_size);
    }

    
    const vec2& normImageFrameSize () const { return m_image_frame_size_norm; }
    void normImageFrameSize (vec2 ns) const { m_image_frame_size_norm = ns; }

    const vec2& normSinglePlotSize () const { return m_single_plot_size_norm; }
    void normSinglePlotSize (vec2 ns) const { m_single_plot_size_norm = ns; }
    
    
    const float& aspectRatio () const { return m_aspect; }
 
    // Trim on all sides
    inline ivec2& trim () const { return m_trim; }
    vec2 trim_norm () { return vec2(trim().x, trim().y) / vec2(m_canvas_size.x, m_canvas_size.y); }
    
    inline ivec2& desired_window_size () const { return m_canvas_size; }
    inline ivec2 canvas_size () { return desired_window_size() - trim() - trim (); }
 
    
    inline void plot_rects (std::vector<Rectf>& plots )
    {
        plots.resize(3);
        auto plot_tl = plots_frame_position_norm();
        auto plot_size = single_plot_size_norm ();
        vec2 plot_vertical (0.0, plot_size.y);
        
        plots[0] = Rectf (plot_tl, vec2 (plot_tl.x + plot_size.x, plot_tl.y + plot_size.y));
        plot_tl += plot_vertical;
        plots[1] = Rectf (plot_tl, vec2 (plot_tl.x + plot_size.x, plot_tl.y + plot_size.y));
        plot_tl += plot_vertical;
        plots[2] = Rectf (plot_tl, vec2 (plot_tl.x + plot_size.x, plot_tl.y + plot_size.y));
        
        vec2 cs (m_canvas_size.x, m_canvas_size.y);
        
        for (Rectf& plot : plots)
        {
            plot.x1 *= cs.x;
            plot.x2 *= cs.x;
            plot.y1 *= cs.y;
            plot.y2 *= cs.y;
            
        }
    }
    

private:
    bool m_isSet;
    
    // image_fram is rect placeholder for image. It will contain the image
    // It is assumed that Image will be on the top left portion
    inline vec2 image_frame_position_norm ()
    {
        vec2 np = vec2 (trim_norm().x, trim_norm().y);
        return np;
    }
    
    inline vec2 image_frame_position ()
    {
        vec2 np = vec2 (image_frame_position_norm().x * desired_window_size().x, image_frame_position_norm().y * desired_window_size().y);
        return np;
    }
    
    inline vec2& image_frame_size_norm (){ return m_image_frame_size_norm;}
    inline ivec2 image_frame_size ()
    {
        static vec2& np = image_frame_size_norm ();
        return ivec2 (np.x * desired_window_size().x, np.y * desired_window_size().y);
    }
    
    // Plot areas on the right
    inline vec2 plots_frame_position_norm ()
    {
        vec2 np = vec2 (1.0 - 2*trim_norm().x - plots_frame_size_norm().x , trim_norm().y);
        return np;
    }
    
    inline vec2 plots_frame_position ()
    {
        vec2 np = vec2 (plots_frame_position_norm().x * desired_window_size().x, plots_frame_position_norm().y * desired_window_size().y);
        return np;
    }
    
    inline vec2 single_plot_size_norm (){ return m_single_plot_size_norm;}
    inline vec2 plots_frame_size_norm (){ vec2 np = vec2 (single_plot_size_norm().x, 3 * single_plot_size_norm().y); return np;}
    
    
    inline ivec2 plots_frame_size () { return ivec2 ((canvas_size().x * plots_frame_size_norm().x),
                                                     (canvas_size().y * plots_frame_size_norm().y)); }
    
    inline ivec2 single_plot_size () { return ivec2 ((canvas_size().x * single_plot_size_norm().x),
                                                     (canvas_size().y * single_plot_size_norm().y)); }
    
    
    inline vec2 canvas_norm_size () { return vec2(canvas_size().x, canvas_size().y) / vec2(getWindowWidth(), getWindowHeight()); }
    inline Rectf text_norm_rect () { return Rectf(0.0, 1.0 - 0.125, 1.0, 0.125); }
    
    
    
    

    mutable Rectf m_window_rect, m_image_rect;
    mutable ivec2 m_canvas_size;
    mutable ivec2 m_trim;
    mutable float m_aspect;
    mutable bool m_keep_aspect;
    mutable vec2 m_image_frame_size_norm;
    mutable vec2 m_single_plot_size_norm;
    
    LayoutSignalWindowSize_t m_windowSizeSignal;
    
    static float aspect (const ivec2& s) { return s.x / (float) s.y; }
    
};

#endif
