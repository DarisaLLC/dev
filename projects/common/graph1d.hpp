
#ifndef _iclip_H_
#define _iclip_H_


#include "cinder/gl/gl.h"
#include "cinder/Color.h"
#include "cinder/Rect.h"
#include "cinder/Function.h"
#include "cinder/gl/Texture.h"
#include "cinder/gl/TextureFont.h"
#include <Iterator>
#include <Functional>

#include "cinder/Timeline.h"

#include "InteractiveObject.h"
#include "async_producer.h"
#include "core/core.hpp"
#include "timeMarker.h"


using namespace std;
using namespace ci;
using namespace ci::app;
class graph1D;

typedef std::shared_ptr<graph1D> Graph1DRef;
typedef std::function<float (float)> Graph1DSetCb;


using    std::numeric_limits;


class graph1D:  public InteractiveObject
{
public:
    
    ci::Font sSmallFont, sBigFont;	// small and large fonts for Text textures
    
    void setFonts ( const Font &smallFont, const Font &bigFont )
    {
        sSmallFont	= smallFont;
        sBigFont	= bigFont;
    }
    
    enum mapping_option
    {
        type_limits = 0,
        type_unit = 1,
        type_signed_unit = 2,
        data_limits = 3
    };
    
    graph1D( std::string name, const ci::Rectf& display_box,
            mapping_option mo = data_limits)
    : InteractiveObject(display_box) , mIsSet(false), m_opt (mo)
    {
        graph1D::setFonts (Font( "Menlo", 18 ), Font( "Menlo", 25 ));
        
        // create label
        //        mTextLayout.clear( cinder::Color::white() ); mTextLayout.setColor( Color(0.5f, 0.5f, 0.5f) );
        //        try { mTextLayout.setFont( Font( "Menlo", 18 ) ); } catch( ... ) { mTextLayout.setFont( Font( "Menlo", 18 ) ); }
        //        TextLayout tmp (mTextLayout);
        //        tmp.addLine( name );
        //        mLabelTex = cinder::gl::Texture::create(tmp.render( true ) );
        
        
    }
    
    // Setup should get called only once.
    // TBD: implement reset
    void setup (size_t length, Graph1DSetCb callback)
    {
        if ( mIsSet ) return;
        mBuffer.resize (length);
        m_CB = bind (callback, std::placeholders::_1);
        mIsSet = true;
    }
    // load the data and bind a function to access it
    void setup (const std::vector<float>& buffer)
    {
        if ( mIsSet ) return;
        mBuffer.clear ();
        std::vector<float>::const_iterator reader = buffer.begin ();
        while (reader != buffer.end())
        {
            mBuffer.push_back (*reader++);
        }
        m_CB = bind (&graph1D::get, this, std::placeholders::_1);
        mIsSet = true;
    }
    
    // load the data and bind a function to access it
    void setup (const trackD1_t& track)
    {
        if ( mIsSet ) return;
        
        const timed_double_vec_t& ds = track.second;
        mBuffer.clear ();
        std::vector<timed_double_t>::const_iterator reader = ds.begin ();
        while (reader != ds.end())
        {
            mBuffer.push_back (reader->second);
            reader++;
        }
        if (m_opt == data_limits)
        {
            auto extr = svl::norm_min_max (mBuffer.begin(), mBuffer.end());
        }
        
        m_CB = bind (&graph1D::get, this, std::placeholders::_1);
        mIsSet = mBuffer.size() == ds.size();
    }
    
    // a NN fetch function using the bound function
    float get (float tnormed) const
    {
        const std::vector<float>& buf = buffer();
        if (empty()) return -1.0;
        int32_t index = floor (tnormed * (buf.size()-1));
        return (index >= 0 && index < buf.size()) ? buf[index] : -1.0f;
    }
    
    void get_marker_position (marker_info& t) const
    {
        t.from_norm (norm_pos().x);
    }
    
    void set_marker_position (marker_info t) const
    {
        std::cout << t.current_frame() << "/" << t.count() << std::endl;
        vec2 pp ((float)t.norm_index(), 0.0);
        norm_pos (pp);

    }
    
    void draw_value_label (float v, float x, float y) const
    {
        TextLayout layout;
        layout.clear( cinder::Color::white() );
        layout.setFont( sSmallFont );
        layout.setColor( Color(0.5f, 0.5f, 0.5f ) );
        
        layout.addLine( to_string(v) );
        auto counter = cinder::gl::Texture::create(layout.render( true ) );
        
        ci::gl::color( ci::Color::white() );
        ci::gl::draw( counter, vec2(x, y));
    }
    
    void draw() const
    {
        
        const Rectf& content = getRect ();
        
        {
            gl::ScopedColor Color( 0.25f, 0.25f, 0.25f);
            ci::gl::drawStrokedRect(content);
        }
        if (! mIsSet) return;
        
        // draw graph
        {
            gl::ScopedColor ColorA( 0.0f, 0.0f, 1.0f, 1.0f );
            gl::begin( GL_LINE_STRIP );
            for( float x = 0; x < content.getWidth(); x ++ )
            {
                float y = m_CB ( x / content.getWidth());
                if (y < 0) continue;
                y = 1.0f - y;
                ci::gl::vertex(vec2( x , y * content.getHeight() ) + content.getUpperLeft() );
            }
            gl::end();
        }
        
        {
            gl::ScopedColor Color( 0.75f, 0.5f, 0.25f );
            glLineWidth(25.f);
            float px = norm_pos().x * content.getWidth();
            float mVal = m_CB (norm_pos().x);
            
            vec2 mid (px, content.getHeight()/2.0);
            mid += content.getUpperLeft();
            
            if (content.contains(mid))
            {
                auto low = content.getUpperLeft().y;
                auto high = low + content.getHeight();
                px += content.getUpperLeft().x;
                ci::gl::drawLine (vec2(px, low), vec2(px, high));
                draw_value_label (mVal, px, mid.y);
                
            }
        }
    }
    const std::vector<float>& buffer () const
    {
        return mBuffer;
    }
    
private:
    
    mutable bool mIsSet;
    mutable float mVal;
    mutable int32_t mIndex;
    std::vector<float>                   mBuffer;
    
    bool empty () const { return mBuffer.empty (); }
    
    cinder::gl::TextureRef						mLabelTex;
    Graph1DSetCb m_CB;
    mapping_option m_opt;
    ci::Anim<marker_info> m_marker;
    
};



#endif // _iclip_H_
