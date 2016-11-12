#include <stdio.h>

#include "ui_contexts.h"
#include "stl_util.hpp"
#include "cinder/app/App.h"
#include "cinder/gl/gl.h"
#include "cinder/Timeline.h"
#include "cinder/Timer.h"
#include "cinder/Camera.h"
#include "cinder/qtime/Quicktime.h"
#include "cinder/params/Params.h"
#include "cinder/ImageIo.h"
#include "qtime_frame_cache.hpp"
#include "CinderOpenCV.h"
#include "Cinder/ip/Blend.h"
#include "opencv2/highgui.hpp"
#include "opencv_utils.hpp"
#include "cinder/ip/Flip.h"


static boost::filesystem::path browseToFolder ()
{
    return Platform::get()->getFolderPath (Platform::get()->getHomeDirectory());
}

void imageDirContext::loadImageDirectory ( const filesystem::path& directory)
{
    m_valid = false;
    
    // clear the list
    mImageFiles.clear();
    
    if( directory.empty() || !filesystem::is_directory( directory ) )
        return;
    
    // make a list of all audio files in the directory
    filesystem::directory_iterator end_itr;
    for( filesystem::directory_iterator i( directory ); i != end_itr; ++i )
    {
        // skip if not a file
        if( !filesystem::is_regular_file( i->status() ) ) continue;
        
        // skip if extension does not match
        string extension = i->path().extension().string();
        extension.erase( 0, 1 );
        if( std::find( mSupportedExtensions.begin(), mSupportedExtensions.end(), extension ) == mSupportedExtensions.end() )
            continue;
        
        // file matches
        mImageFiles.push_back( i->path() );
    }
    
    std::sort (mImageFiles.begin(), mImageFiles.end(), stl_utils::naturalPathComparator());
    
 }

imageDirContext::imageDirContext(ci::app::WindowRef window) : uContext(window)
{
    setup ();
    if (is_valid())
    {
        
        mCbMouseDrag = mWindow->getSignalMouseMove().connect( std::bind( &imageDirContext::mouseMove, this, std::placeholders::_1 ) );
        mCbKeyDown = mWindow->getSignalKeyDown().connect( std::bind( &clipContext::keyDown, this, std::placeholders::_1 ) );
//        getWindow()->setTitle( mPath.filename().string() );
    }
    
}

bool imageDirContext::is_valid ()
{
    return m_valid;
}
void imageDirContext::setup()
{
    m_valid = false;
    mType = Type::image_dir_viewer;

    
    // make a list of valid audio file extensions and initialize audio variables
    const char* extensions[] = { "jpg", "png", "JPG" };
    mSupportedExtensions = vector<string>( extensions, extensions + 3 );
    
    // Browse for the image folder
    auto folderPath = browseToFolder ();
    
    loadImageDirectory(folderPath);
    
    if ( mImageFiles.empty () ) return;

    mTotalItems = mImageFiles.size();
    mItemExpandedWidth = 500;
    mItemRelaxedWidth = 848 / mTotalItems;
    mItemHeight = 564;

    float xPos = 0;

    std::vector<filesystem::path>::const_iterator pItr = mImageFiles.begin();
    for (; pItr < mImageFiles.end(); pItr++)
    {
        mItems.push_back( AccordionItem( timeline(),
                                        xPos,
                                        0,
                                        mItemHeight,
                                        mItemRelaxedWidth,
                                        mItemExpandedWidth,
                                        gl::Texture::create( loadImage( pItr->string() ) ),
                                        pItr->parent_path().string(),
                                        pItr->filename().string()));
        xPos += mItemRelaxedWidth;
    }

    
    // similar to mCurrentSelection = null;
    mCurrentSelection = mItems.end();
}


void imageDirContext::mouseMove( MouseEvent event )
{
    list<AccordionItem>::iterator mNewSelection = mItems.end();
    
    for( list<AccordionItem>::iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
        if( itemIt->isPointIn( event.getPos() ) ) {
            mNewSelection = itemIt;
            break;
        }
    }
    
    if( mNewSelection != mCurrentSelection) {
        float xPos = 0;
        float contractedWidth = (mTotalItems*mItemRelaxedWidth - mItemExpandedWidth)/float(mTotalItems - 1);
        mCurrentSelection = mNewSelection;
        
        if (mCurrentSelection == mItems.end()) {
            for( list<AccordionItem>::iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
                itemIt->animTo(xPos, mItemRelaxedWidth);
                xPos += mItemRelaxedWidth;
            }
        } else {
            for( list<AccordionItem>::iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
                if( itemIt == mCurrentSelection ) {
                    itemIt->animTo(xPos, mItemExpandedWidth, true);
                    xPos += mItemExpandedWidth;
                } else {
                    itemIt->animTo(xPos, contractedWidth);
                    xPos += contractedWidth;
                }
            }
        }
    }
}

void imageDirContext::update()
{
    for( list<AccordionItem>::iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
        itemIt->update();
    }
}

void imageDirContext::draw()
{	
    gl::clear( Color( 1, 1, 1 ) );
    gl::enableAlphaBlending();
    
    for( list<AccordionItem>::iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
        itemIt->draw();
    }
}
