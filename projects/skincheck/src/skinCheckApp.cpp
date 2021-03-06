#include "skinCheck.h"

void skinCheckApp::prepareSettings( skinCheckApp::Settings *settings )
{
    // By default, multi-touch is disabled on desktop and enabled on mobile platforms.
    // You enable multi-touch from the SettingsFn that fires before the app is constructed.
    settings->setMultiTouchEnabled( true );
    
    // On mobile, if you disable multitouch then touch events will arrive via mouseDown(), mouseDrag(), etc.
    //	settings->setMultiTouchEnabled( false );
}

int skinCheckApp::which (vec2 position)
{
    vector<float> dists;
    float squareDist = mActiveRadius*mActiveRadius;
    for (unsigned tt = 0; tt < mCenters.size(); tt++)
    {
        const touch& cc = mCenters[tt];
        dists.push_back((position.x - cc.x) * (position.x - cc.x) +
                        (position.y - cc.y) * (position.y - cc.y));
    }
    vector<float>::iterator mini = std::min_element(dists.begin(), dists.end());
    if (*mini > squareDist) return -1;
    auto index = std::distance(dists.begin(),mini);
    return index;
    
}


void skinCheckApp::setup()
{
    printDevices();
    float x_offset = 0.92;
    float y_offset = 0.25;
    float y_step = 0.12;
    mActiveRadius = y_step * .45;
    
    mTouchPos.x = mTouchPos.y = -1;
    
#if not defined( CINDER_COCOA_TOUCH )
    cv::namedWindow ("window", cv::WINDOW_AUTOSIZE);
#endif
    
    mCenters.emplace_back(vec2(x_offset, y_offset+=y_step), ColorAf(1.0f,0.0f,0.0f),mActiveRadius);
    mCenters.emplace_back(vec2(x_offset,y_offset+=y_step), ColorAf(1.0f,0.7f,0.0f),mActiveRadius);
    mCenters.emplace_back(vec2(x_offset,y_offset+=y_step), ColorAf(0.0f,1.0f,0.0f),mActiveRadius);
    mCenters.emplace_back(vec2(x_offset,y_offset+=y_step), ColorAf(0.0f,0.0f,1.0f),mActiveRadius);
    //    mCenters.emplace_back(vec2(x_offset,y_offset+=y_step), ColorAf(0.5f,0.5f,0.5f),mActiveRadius);
    
    
    for (unsigned cc = 0; cc < mCenters.size(); cc++)
    {
        mCenters[cc].set_alive_call_back(static_cast<touch::callback_t>(std::bind(&skinCheckApp::set_check, this, cc, true)));
        mCenters[cc].set_died_call_back(static_cast<touch::callback_t>(std::bind(&skinCheckApp::set_check, this, cc, false)));
    }
    
    
    mFocusRects.emplace_back(0.25f,0.25f,0.75f,0.75f);
    mPad = 0.25;
    
    overex_producer = std::bind(&skinCheckApp::overex_producer_impl, this);
    shadow_producer = std::bind(&skinCheckApp::shadow_producer_impl, this);
    edge_producer = std::bind(&skinCheckApp::edge_producer_impl, this);
    blur_producer = std::bind(&skinCheckApp::blur_producer_impl, this);
    
    mProducerMap[0] = overex_producer;
    mProducerMap[1] = shadow_producer;
    mProducerMap[2] = edge_producer;
    mProducerMap[3] = blur_producer;
    
    assert(mChecks.size() >= mProducerMap.size());
    mChecks.reset();
    
    
#if defined( CINDER_COCOA_TOUCH )
    mFont = Font( "Cochin-Italic", 18 );
#elif defined( CINDER_COCOA )
    mFont = Font( "BigCaslon-Medium", 18 );
#else
    mFont = Font( "Times New Roman", 18 );
#endif
    mTextureFont = gl::TextureFont::create( mFont );
    
    
    mShouldQuit = false;
    mTextures = std::unique_ptr< ConcurrentCircularBuffer<gl::TextureRef> > (new ConcurrentCircularBuffer<gl::TextureRef>( 5 )); // room for 5 images
    // create and launch the thread with a new gl::Context just for that thread
    gl::ContextRef backgroundCtx = gl::Context::create( gl::context() );
    mThread = shared_ptr<thread>( new thread( bind( &skinCheckApp::captureThreadFn, this, backgroundCtx ) ) );
    mLastTime = getElapsedSeconds() - 10; // force an initial update by make it "ten seconds ago"
    gl::enableAlphaBlending();
    
}


void skinCheckApp::cleanup()
{

}

skinCheckApp::~skinCheckApp()
{
    mShouldQuit = true;
    mTextures->cancel();
    mThread->join();
}


void skinCheckApp::captureThreadFn(gl::ContextRef context)
{
    ci::ThreadSetup threadSetup; // instantiate this if you're talking to Cinder from a secondary thread
    // we received as a parameter a gl::Context we can use safely that shares resources with the primary Context
    context->makeCurrent();
    
    try {
        mCapture = Capture::create( 816, 612);
        
        mCapture->start();
    }
    catch( ci::Exception &exc ) {
        CI_LOG_EXCEPTION( "Failed to init capture ", exc );
    }
    mCount = 0;
    mPos = getWindowCenter();
    
   	while(! mShouldQuit )
    {
        if( mCapture && mCapture->checkNewFrame() )
        {
            // we need to wait on a fence before alerting the primary thread that the Texture is ready
#if ! defined( CINDER_GL_ES ) || defined( CINDER_GL_ES_3 )
            auto fence = gl::Sync::create();
#else
            mFence = glFenceSyncAPPLE(GL_SYNC_GPU_COMMANDS_COMPLETE_APPLE, 0);
#endif
           	mSurface = mCapture->getSurface();
            auto tex = gl::Texture::create(*mSurface, gl::Texture::Format().loadTopDown() );
            mBGR = toOcv(*mSurface);
            cv::normalize(mBGR,mBGR,0.0, 255.0, cv::NORM_MINMAX);
            
            cv::split(mBGR, mB_G_R);
           	mGray =  toOcv( Channel( *mSurface ));
            mScores.clear();
            assert (mChecks.count() <= 1);
            for (unsigned cc = 0; cc < mChecks.size(); cc++)
                if (mChecks[cc])
                    mProducerMap[cc]();
            
            
            // wait on the fence
#if ! defined( CINDER_GL_ES ) || defined( CINDER_GL_ES_3 )
            fence->clientWaitSync();
#else
            glClientWaitSyncAPPLE(mFence, GL_SYNC_FLUSH_COMMANDS_BIT_APPLE, GL_TIMEOUT_IGNORED_APPLE);
#endif
            mTextures->pushFront( tex );
            mCount++;
        }
    }
    
}

void skinCheckApp::clear_overlay()
{
    mOverlay = gl::TextureRef ();
}
void skinCheckApp::update()
{
    if (! update_touch() ) clear_overlay();
    
    double timeSinceLastImage = getElapsedSeconds() - mLastTime;
    if( ( timeSinceLastImage > 0.001 ) && mTextures->isNotEmpty() )
    {
        mLastTexture = mTexture; // the "last" texture is now the current text
        mTextures->popBack( &mTexture );
        mLastTime = getElapsedSeconds();
    }
}


void skinCheckApp::draw()
{
    render_dynamic_content();
    render_static_content();
    
}

/*
 * Blur is computed by measuring how fast the auto-correlation function drops.
 * Auto-correlation function is approximated by computing normalized correlation
 * of the ROI described by - and | within the ROI defined by *, together we can call them
 * the blur probe region.
 * We can have many blur probe regions. This prototype implements just one.
 *
 *      o  ______________________________________________ X-axis
 *        |
 *        |  Y-axis
 *        |                *****************************
 *        |                *                            *
 *        |                *      ---------------       *
 *        |                *      |              |      *
 *        |                *      |              |      *
 *        |                *      |              |      *
 *        |                *      |              |      *
 *        |                *      |              |      *
 *        |                *      |              |      *
 *        |                *      ---------------       *
 *        |                *                            *
 *        |                *****************************
 *
 *
 *
 */


void skinCheckApp::compute_blur(const cv::Mat& src, const std::vector<Area>& rois, const int pad, std::vector<float>& scores)
{
    if (src.ptr() == nullptr) return;
    for (Area focus_rect : rois)
    {
        Area focus_inner (focus_rect.getX1() + pad, focus_rect.getY1() + pad, focus_rect.getLR().x - pad , focus_rect.getLR().y - pad);
        cv::Mat focus = src (toOcv(focus_rect));
        cv::Mat inner = src (toOcv(focus_inner));
        ivec2 match = focus_rect.getSize() - focus_inner.getSize() + ivec2(1,1);
        
        cv::Mat space (match.x, match.y,CV_32FC1);
        matchTemplate(focus, inner, space, CV_TM_CCORR_NORMED);
        double min_c = -1.0;;
        cv::minMaxIdx(space, &min_c);
        min_c = std::signbit(min_c) ? 0 : std::sqrt(1.0 - min_c);
        scores.push_back(min_c);
    }
    
}

void skinCheckApp::blur_producer_impl ()
{
    std::vector<Area> focusAreas;
    vec2 minSize;
    int maxSize = getWindowWidth() * getWindowHeight();
    for (Rectf aa : mFocusRects)
    {
        Area ia;
        norm2image(aa, ia);
        focusAreas.push_back(ia);
        int size = ia.getWidth()*ia.getHeight();
        if (size < maxSize)
        {
            maxSize = size;
            minSize = ia.getSize();
        }
    }
    
    compute_blur(mGray, focusAreas, mPad * minSize.x, mScores);
    if(mScores.size())
    {
        float product = std::accumulate(mScores.begin(), mScores.end(), 1.0, std::multiplies<float>());
        float oneOver = 1.0f / ((float)mScores.size());
        mScore = std::pow(product, oneOver);
    }
    
}

void skinCheckApp::overex_producer_impl()
{
    cv::Mat mask;
    std::vector<cv::Mat> single;
    single.push_back(mGray);
    compute_overex_overlay(single, mask);
    mOverlay = gl::Texture::create(fromOcv(mask), gl::Texture::Format().loadTopDown());
}

void skinCheckApp::shadow_producer_impl()
{
    cv::Mat mask;
    compute_shadow_overlay(mB_G_R, mask);
    mOverlay = gl::Texture::create(fromOcv(mask), gl::Texture::Format().loadTopDown());
}

void skinCheckApp::edge_producer_impl()
{
    cv::Mat mask;
    std::vector<cv::Mat> single;
    single.push_back(mGray);
    compute_edge_overlay(single, mask);
    mOverlay = gl::Texture::create(fromOcv(mask), gl::Texture::Format().loadTopDown());
}


void skinCheckApp::update_overlay( cv::Mat& mask)
{
    //            compute_overex_overlay(mB_G_R, mask);
}

void skinCheckApp::compute_edge_overlay(const std::vector<cv::Mat> &gray_one, cv::Mat& mask)
{
    assert(gray_one.size() == 1);
    cv::GaussianBlur(gray_one[0], mask, cv::Size(5,5), 2.0f);
    cv::Canny(mask, mask, 30, 20 * 5, 7);
    cv::normalize(mask,mask, 0.0, 255.0, cv::NORM_MINMAX);
}

void skinCheckApp::compute_shadow_overlay(const std::vector<cv::Mat>& bgr, cv::Mat& mask)
{
    assert(bgr.size()==3 || bgr.size()==4);
    // Compute linearity and equlivance of color channels
    absdiff(bgr[0]+bgr[2],bgr[1]*2, mask);
    cv::normalize(mask,mask, 0.0, 255.0, cv::NORM_MINMAX);
}


void skinCheckApp::compute_overex_overlay(const std::vector<cv::Mat>& gray_one, cv::Mat& mask)
{
    assert(gray_one.size()==1);
    cv::threshold(gray_one[0], mask, 254,255, CV_THRESH_BINARY);
    cv::normalize(mask,mask, 0.0, 255.0, cv::NORM_MINMAX);
}


void skinCheckApp::compute_skin_overlay(const std::vector<cv::Mat>& bgr, cv::Mat& mask)
{
    assert(bgr.size()==3);
    
}

std::string skinCheckApp::pixelInfo( const ivec2 &position )
{
    // first, specify a small region around the current cursor position
    float scaleX = mTexture->getWidth() / (float)getWindowWidth();
    float scaleY = mTexture->getHeight() / (float)getWindowHeight();
    ivec2	pixel( (int)( position.x * scaleX ), (int)( (position.y * scaleY ) ) );
    auto val = mSurface->getPixel(pixel);
    ostringstream oss( ostringstream::ate );
    oss << "[" << pixel.x << "," << pixel.y << "]";
    oss << to_string((int)val.r) << "," << to_string((int)val.g) << "," << to_string((int)val.b);
    
    return oss.str();
    
}


void skinCheckApp::render_dynamic_content()
{
    gl::clear();
    
    {
        gl::ScopedModelMatrix modelScope;
        Rectf bounds = getWindowBounds();
        
#if defined( CINDER_COCOA_TOUCH )
        // change iphone to landscape orientation
        gl::rotate( M_PI / 2 );
        gl::translate( 0, - getWindowWidth() );
        
        bounds = Rectf ( 0, 0, getWindowHeight(), getWindowWidth() );
#endif
        if( mTexture )
        {
            gl::ScopedModelMatrix modelScope;
            gl::ScopedBlendAlpha blend;
            gl::ScopedColor color (ColorA (1.0f,0.0f,0.0f,0.50f));
            
            gl::draw(mOverlay, bounds);
            
            {
                gl::ScopedColor color (ColorA (1.0,1.0,1.0,0.50f));
                gl::draw( mTexture, bounds );
                
            }
        }
    }
}


void skinCheckApp::render_static_content()
{
    
    {
        gl::ScopedBlendAlpha sAb;
        std::for_each(mCenters.begin(), mCenters.end(), [this](touch& tt) {tt.draw(getWindowBounds());} );
        
        
        ostringstream oss( ostringstream::ate );
        oss << fixed << setprecision(3);
        
        oss.str( "[" );
        oss << (int) getAverageFps();
        oss << ",";
        oss << mCount;
        oss << ",";
        oss << mTextures->getSize();
        oss << "]";
        mTextureFont->drawString( oss.str(), ivec2( getWindowWidth()*.8, getWindowHeight()*.95));
        
        if (mChecks[3])
        {
            oss.str( "Blur result: " );
            oss << mScore;
            mTextureFont->drawString( oss.str(), ivec2( 10, 65 ) );
        }
        
        if (mSurface && getWindowBounds().contains(mTouchPos))
        {
            gl::pushModelView();
            Rectf frac (0.60f, 0.05f, 0.98f, 0.10f);
            Rectf box (frac.x1*getWindowWidth(),frac.y1*getWindowHeight(),
                       frac.x2*getWindowWidth(),frac.y2*getWindowHeight());
            gl::drawSolidRoundedRect (box, 0.5f);
            gl::ScopedColor color (ColorA (0.0f,1.0f,1.0,1.0f));
            mTextureFont->drawString( pixelInfo(mTouchPos), ivec2( box.x1, (box.y1+box.y2)/2 ) );
            gl::popModelView();
        }
        
        
    }
}

bool skinCheckApp::check_touch(const vec2& tpos)
{
    vec2 normed;
    image2norm (tpos, normed);
    auto ww = which(normed);
    if (ww == -1) return false;
    for (int cc = 0; cc < mCenters.size(); cc++)
        mCenters[cc].update(cc == ww);
    return true;
    
}

bool skinCheckApp::update_touch ()
{
    int tc = 0;
#if not defined( CINDER_MAC )
    for( const auto &touch : getActiveTouches() )
    {
        mTouchPos = touch.getPos();
#endif
        bool ok = getWindowBounds().contains(mTouchPos) && check_touch(mTouchPos);
        tc += ok;
#if not defined( CINDER_MAC )
    }
#endif
    return tc > 0;
}





void skinCheckApp::printDevices()
{
    for( const auto &device : Capture::getDevices() ) {
        console() << "Device: " << device->getName() << " "
#if defined( CINDER_COCOA_TOUCH )
        << ( device->isFrontFacing() ? "Front" : "Rear" ) << "-facing"
#endif
        << endl;
    }
}



CINDER_APP( skinCheckApp, RendererGl );
