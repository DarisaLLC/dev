#include "DirMovie.h"
#include "cinder/app/App.h"
#include "cinder/Utilities.h"
#include "cinder/CinderMath.h"

using namespace std;
using namespace ci;

/*******************************************************************************
 * Construction
 */

DirMovie::DirMovie( const fs::path &directory, const std::string &extension, const double fps ) :
mThreadIsRunning( false ),
mDataIsFresh( false ),
mLoopEnabled( true ),
mAverageFps( 0 ),
mFpsLastSampleTime( 0 ),
mFpsFrameCount( 0 ),
mFpsLastFrameCount( 0 ),
mFrameRate( fps ),
mNextFrameTime( app::getElapsedSeconds() ),
mInterruptTriggeredFoRealz( false ),
mCurrentFrameIdx( 0 ),
mCurrentFrameIsFresh( false ),
mNumFrames( 0 )
{
    setPlayRate( 1.0 );

    using namespace ci::fs;

    if ( !exists( directory ) ) throw LoadError( directory.string() + " does not exist" );
    if ( !is_directory( directory ) ) throw LoadError( directory.string() + " is not a directory" );


    mThreadData.extension       = extension;
    mThreadData.directoryPath   = directory;

    readFramePaths();

    mThread                     = thread( bind( &DirMovie::updateFrameThreadFn, this ) );
    mThreadIsRunning            = true;
}

DirMovie::~DirMovie()
{
    mThreadIsRunning = false;
    mThread.join();
}


/*******************************************************************************
 * Exception Handling
 */

void
DirMovie::warn( const string &warning )
{
    cout << warning << endl;
}

/*******************************************************************************
 * Lifecycle
 */

void
DirMovie::update()
{
    if ( mDataIsFresh )
    {
        {
            lock_guard< mutex > lock( mMutex );
            mTexture = ci::gl::Texture::Dds( mThreadData.buffer->createStream(), ::mdds::Texture::Format() );
        }

        if ( mTexture == nullptr ) warn( "error creating texture" );
        	
        mDataIsFresh = false;
    }
}

void
DirMovie::draw()
{
    if ( mTexture ) gl::draw( mTexture );
}

/*******************************************************************************
 * Play control
 */

double
DirMovie::getFrameRate() const
{
    return mFrameRate;
}

void
DirMovie::updateAverageFps()
{
    double now = app::getElapsedSeconds();
    mFpsFrameCount++;
    
    if( now > mFpsLastSampleTime + app::App::get()->getFpsSampleInterval() ) {
        //calculate average Fps over sample interval
        uint32_t framesPassed = mFpsFrameCount - mFpsLastFrameCount;
        mAverageFps = framesPassed / (now - mFpsLastSampleTime);
        mFpsLastSampleTime = now;
        mFpsLastFrameCount = mFpsFrameCount;
    }
}

double
DirMovie::getAverageFps() const
{
    return mAverageFps;
}

void
DirMovie::setPlayRate( const double newRate )
{
    mInterruptTriggeredFoRealz = newRate != mPlayRate;
    mPlayRate = newRate;

    if ( mInterruptTriggeredFoRealz )
    {
        lock_guard< mutex > lock( mMutex );
        mInterruptFrameRateSleepCv.notify_all();
    }
}

double
DirMovie::getPlayRate() const
{
    return mPlayRate;
}

void
DirMovie::readFramePaths()
{
    using namespace ci::fs;

    mThreadData.framePaths.clear();

    for ( auto it = directory_iterator( mThreadData.directoryPath ); it != directory_iterator(); it++ )
    {
        if ( it->path().extension() != mThreadData.extension ) continue;

        mThreadData.framePaths.push_back( it->path() );
    }

    mNumFrames = mThreadData.framePaths.size();
}

void
DirMovie::seekToTime( const double seconds )
{
    seekToFrame( seconds * mFrameRate );
}

void
DirMovie::seekToFrame( const size_t frame )
{
    mCurrentFrameIdx = frame;
    mCurrentFrameIsFresh = true;
    mInterruptTriggeredFoRealz = true;
    lock_guard< mutex > lock( mMutex );
    mInterruptFrameRateSleepCv.notify_all();
}

void
DirMovie::seekToStart()
{
    seekToFrame( 0 );
}

void
DirMovie::seekToEnd()
{
    seekToFrame( -1 );
}

size_t
DirMovie::getCurrentFrame() const
{
    return mCurrentFrameIdx;
}

size_t
DirMovie::getNumFrames() const
{
    return mNumFrames;
}

double
DirMovie::getCurrentTime() const
{
    return (double)mCurrentFrameIdx / mFrameRate;
}

double
DirMovie::getDuration() const
{
    return (double)mNumFrames / mFrameRate;
}

/*******************************************************************************
 * Async
 */

void
DirMovie::updateFrameThreadFn()
{
    using namespace ci::fs;

    ci::ThreadSetup threadSetup;

    while ( mThreadIsRunning )
    {
        mNextFrameTime = app::getElapsedSeconds();
        
        unique_lock< mutex > lock( mMutex );

        // Read data into memory buffer
        auto ds_path = DataSourcePath::create( mThreadData.framePaths[ mCurrentFrameIdx ] );
        mThreadData.buffer = DataSourceBuffer::create( ds_path->getBuffer() );


        mDataIsFresh = true;
        updateAverageFps();

        nextFramePosition();


        // FrameRate control, cribbed from AppImplMswBasic.cpp
        double currentSeconds   = app::getElapsedSeconds();
        double secondsPerFrame  = mPlayRate == 0.0 ? 1.0 : ((1.0 / math< double >::abs( mPlayRate )) / mFrameRate);
        mNextFrameTime          = mNextFrameTime + secondsPerFrame;
        if ( mNextFrameTime > currentSeconds )
        {
            int ms = (mNextFrameTime - currentSeconds) * 1000.0;
            mInterruptFrameRateSleepCv.wait_for( lock,
                                                 chrono::milliseconds( ms ),
                                                 [&]{ return mInterruptTriggeredFoRealz; } );
        }
        mInterruptTriggeredFoRealz = false;
    }
}

/*******************************************************************************
 * Position control
 */

void
DirMovie::nextFramePosition()
{
    using namespace ci::fs;

    if ( !mCurrentFrameIsFresh )
        mCurrentFrameIdx += mPlayRate == 0.0 ? 0 : (mPlayRate > 0 ? 1 : -1);

    mCurrentFrameIsFresh = false;

    if ( mCurrentFrameIdx == mNumFrames )
    {
        if ( mLoopEnabled ) mCurrentFrameIdx = 0;
        else mCurrentFrameIdx = mNumFrames - 1;
    }
    else if ( mCurrentFrameIdx == (size_t)-1 )
    {
        if ( mLoopEnabled ) mCurrentFrameIdx = mNumFrames -1;
        else mCurrentFrameIdx = 0;
    }
}