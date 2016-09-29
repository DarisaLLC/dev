#include "cinder/app/App.h"
#include "cinder/Camera.h"
#include "cinder/params/Params.h"

#include <functional>

using namespace ci;
using namespace ci::app;

class cVisibleApp : public App {
public:
    void prepareSettings( Settings *settings ) { settings->enableHighDensityDisplay(); }
    void setup();
    void resize();
    void draw();
    void button();
    
private:
    CameraPersp				mCam;
    params::InterfaceGlRef	mParams;
    float					mObjSize;
    quat					mObjOrientation;
    ColorA					mColor;
    std::string				mString;
    
    void					setLightDirection( vec3 direction );
    vec3					getLightDirection() { return mLightDirection; }
    vec3					mLightDirection;
    uint32_t				mSomeValue;
};

void cVisibleApp::setLightDirection( vec3 direction )
{
    console() << "Light direction: " << direction << std::endl;
    mLightDirection = direction;
}

void cVisibleApp::setup()
{
    mObjSize = 4;
    mLightDirection = vec3( 0, 0, -1 );
    mColor = ColorA( 0.25f, 0.5f, 1, 1 );
    mSomeValue = 2;
    
    // Setup our default camera, looking down the z-axis
    mCam.lookAt( vec3( -20, 0, 0 ), vec3::zero() );
    
    // Create the interface and give it a name.
    mParams = params::InterfaceGl::create( getWindow(), "App parameters", toPixels( Vec2i( 200, 400 ) ) );
    
    // Setup the parameters
    mParams->addParam( "Cube Size", &mObjSize ).min( 0.1f ).max( 20.5f ).keyIncr( "z" ).keyDecr( "Z" ).precision( 2 ).step( 0.02f );
    mParams->addParam( "Cube Rotation", &mObjOrientation );
    mParams->addParam( "Cube Color", &mColor );
    mParams->addParam( "String ", &mString );
    
    mParams->addSeparator();
    
    // Attach a callback that is fired after a target is updated.
    mParams->addParam( "some value", &mSomeValue ).updateFn( [this] { console() << "new value: " << mSomeValue << std::endl; } );
    
    // Add a param with no target, but instead provide setter and getter functions.
    std::function<void( vec3 )> setter	= std::bind( &cVisibleApp::setLightDirection, this, std::placeholders::_1 );
    std::function<vec3 ()> getter		= std::bind( &cVisibleApp::getLightDirection, this );
    mParams->addParam( "Light Direction", setter, getter );
    
    // Other types of controls that can be added to the interface
    mParams->addButton( "Button!", std::bind( &cVisibleApp::button, this ) );
    mParams->addText( "text", "label=`This is a label without a parameter.`" );
}

void cVisibleApp::button()
{
    app::console() << "Clicked!" << std::endl;
    mParams->setOptions( "text", "label=`Clicked!`" );
}

void cVisibleApp::resize()
{
    mCam.setAspectRatio( getWindowAspectRatio() );
}

void cVisibleApp::draw()
{
    // this pair of lines is the standard way to clear the screen in OpenGL
    gl::enableDepthRead();
    gl::enableDepthWrite();
    gl::clear( Color::gray( 0.1f ) );
    
    glLoadIdentity();
    glEnable( GL_LIGHTING );
    glEnable( GL_LIGHT0 );
    GLfloat lightPosition[] = { -mLightDirection.x, -mLightDirection.y, -mLightDirection.z, 0 };
    glLightfv( GL_LIGHT0, GL_POSITION, lightPosition );
    glMaterialfv( GL_FRONT, GL_DIFFUSE,	mColor );
    
    gl::setMatrices( mCam );
    gl::rotate( mObjOrientation );
    gl::color( mColor );
    gl::drawCube( vec3::zero(), vec3( mObjSize, mObjSize, mObjSize ) );
    
    // Draw the interface
    mParams->draw();
}

CINDER_APP_BASIC( cVisibleApp, RendererGl )