/*
 * Code Copyright 2011 Darisa LLC
 */



#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-private-field"
#pragma GCC diagnostic ignored "-Wcomma"

#include "VisibleApp.h"
using namespace cinder;
namespace ui=ImGui;



namespace VisibleAppControl{
    /**
     When this is set to false the threads managed by this program will stop and join the main thread.
     */
   // bool ThreadsShouldStop = false;
}

static void ShowHelpMarker(const char* desc)
{
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(450.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void prepareSettings( App::Settings *settings )
{
    //settings->setHighDensityDisplayEnabled();
    settings->setWindowPos(0, 0);
    settings->setFrameRate(10);
    //Prevent the computer from dimming the display or sleeping
    settings->setPowerManagementEnabled(false);
}


void VisibleApp::dispatch_lif_viewer (const int serie_index)
{
    auto name_index_itr = mLifRef->index_to_name_map().find(serie_index);
    static std::string space ("  ");
    vlogger::instance().console()->info(" Serie " + svl::toString(serie_index) + " Requested " );
    if (name_index_itr != mLifRef->index_to_name_map().end())
    {
        vlogger::instance().console()->info(" Serie " + svl::toString(serie_index) + " Fetched " );
        const std::string& name = name_index_itr->second;
        assert(exists(mRunAppPath));
        auto command = mRunAppPath.string()  + space  + " --args " + mCurrentLifFilePath.string() + space + name;
        if (m_isIdLabLif) command += space + vac::LIF_CUSTOM;
        
        command = "open -n -F -a " + command;
        vlogger::instance().console()->info(command);
        std::system(command.c_str());
        
    }else
        vlogger::instance().console()->error(" Serie " + svl::toString(serie_index) + " Could not be Fetched " );
}


void VisibleApp::windowMouseDown( MouseEvent &mouseEvt )
{
    //  update_log ( "Mouse down in window" );
}


void VisibleApp::fileDrop( FileDropEvent event )
{
    auto imageFile = event.getFile(0);
    
    const fs::path& some_path = imageFile;
    mFileExtension = some_path.extension().string();
    mFileName = some_path.filename().string();
    
    if (! exists(some_path) ) return;
    
    
    if (mFileExtension.compare(".lif") == 0)
        initData(some_path);
    else{
        std::string msg = some_path.string() + " is not a valid path to a file ";
        vlogger::instance().console()->info(msg);
    }
}


bool VisibleApp::shouldQuit()
{
    return true;
}

void VisibleApp::fileOpen (){
    
    auto some_path = getOpenFilePath(); //"", extensions);
    mFileExtension = some_path.extension().string();
    mFileName = some_path.filename().string();
    
    //   boost::filesystem::path some_path = "/Users/arman/Pictures/lif/3channels.lif";
    if (! some_path.empty() || exists(some_path))
        initData(some_path);
    else{
        std::string msg = some_path.string() + " is not a valid path to a file ";
        vlogger::instance().console()->info(msg);
    }
    
}
//
void VisibleApp::setup()
{

    ui::Initialize(ui::Options());
    
    // StyleColorsLightGreenImGuiStyle* st = &ImGui::GetStyle();
    // ImGui::StyleColorsLightGreen(st);
    
    fs::path root_output_dir = VisibleAppControl::get_visible_app_support_directory();
    VisibleAppControl::setup_loggers( root_output_dir, app_log, mFileName);
    
    const fs::path& appPath = ci::app::getAppPath();
    mRunAppPath = appPath / "Visible.app/Contents/MacOS/VisibleRun.app";
    mRunAppAppString = mRunAppPath.string();
    if(exists(mRunAppPath))vlogger::instance().console()->info(mRunAppPath.string()  + " Exists ");
    else
        vlogger::instance().console()->error(mRunAppAppString  + " is incorrect or App content is damaged ");
    
    
    const fs::path plist = appPath / "Visible.app/Contents/Info.plist";
    std::ifstream stream(plist.c_str(), std::ios::binary);
    Plist::readPlist(stream, mPlist);
    if(mVisibleScope== nullptr){
        mVisibleScope = gl::Texture::create( loadImage( loadResource(VISIBLE_SCOPE  )));
    }
    
    setWindowSize(APP_WIDTH/2,APP_HEIGHT/2);
    setFrameRate( 60 );
    setWindowPos(getWindowSize()/4);

    WindowRef ww = getWindow ();
    if( mVisibleScope ){
        gl::draw( mVisibleScope, getWindowBounds() );
    }
    
    ww->getRenderer()->makeCurrentContext(true);
    std::string buildN =  boost::any_cast<const string&>(mPlist.find("CFBundleVersion")->second);
    ww->setTitle ("Visible ( build: " + buildN + " ) ");
    mFont = Font( "Menlo", 18 );
    mSize = vec2( getWindowWidth(), getWindowHeight() / 12);
    
    // fonts
    Font smallFont = Font( "TrebuchetMS-Bold", 16 );
    Font medFont = Font( "TrebuchetMS-Bold", 32 );
    Font bigFont   = Font( "TrebuchetMS-Bold", 72 );
    Item::setFonts( smallFont, bigFont );
    {
        TextLayout layout;
        layout.clear( ColorA( 1, 1, 1, 0 ) );
        layout.setFont( medFont );
        Color  color_choice ( 166/255.,17/255.,177/255. );
        layout.setColor( color_choice );
        layout.addCenteredLine("Drag & Drop");
        layout.addCenteredLine("a LIF File");
        mDropFileTextTexture = gl::Texture::create( layout.render( true ) );
    }
    
    // title text
    TextLayout layout;
    layout.clear( ColorA( 1, 1, 1, 0 ) );
    layout.setFont( bigFont );
    layout.setColor( Color::white() );
    layout.addLine( "Icelandic Colors" );
    mTitleTex		= gl::Texture2d::create( layout.render( true ) );
    
    // positioning
    mLeftBorder		= 50.0f;
    mTopBorder		= 375.0f;
    mItemSpacing	= 22.0f;
    
 
    
    // create items
    std::vector<std::string> extensions = {"lif"};
    auto platform = ci::app::Platform::get();
    
    auto home_path = platform->getHomeDirectory();
    vlogger::instance().console()->info(home_path.string());
    
    
    // textures and colors
    mFgImage		= gl::Texture2d::create( APP_WIDTH / 2, APP_HEIGHT / 2 );
    mBgImage		= gl::Texture2d::create( APP_WIDTH / 2, APP_HEIGHT / 2 );
    mFgAlpha		= 1.0f;
    mBgColor		= Color::white();
    
    // swatch graphics (square and blurry square)
    auto palette_path = platform->getResourceDirectory();
    std::string small =  "swatchSmall.png";
    std::string large =  "swatchLarge.png";
    auto small_path = palette_path / small ;
    mSwatchSmallTex	= gl::Texture2d::create( loadImage( small_path ) );
    auto large_path = palette_path / large ;
    mSwatchLargeTex	= gl::Texture2d::create( loadImage( large_path ) );
    
    // state
    if(! mItems.empty()){
        mMouseOverItem		= mItems.end();
        mNewMouseOverItem	= mItems.end();
        mSelectedItem		= mItems.begin();
        mSelectedItem->select( timeline(), mLeftBorder );
    }
    
    getSignalShouldQuit().connect( std::bind( &VisibleApp::shouldQuit, this ) );
    
    getWindow()->getSignalMove().connect( std::bind( &VisibleApp::windowMove, this ) );
    getWindow()->getSignalDisplayChange().connect( std::bind( &VisibleApp::displayChange, this ) );
    getWindow()->getSignalDraw().connect([&]{draw();});
    getWindow()->getSignalClose().connect(std::bind( &VisibleApp::windowClose, this) );
    getWindow()->getSignalResize().connect(std::bind( &VisibleApp::resize, this) );
    
    getSignalDidBecomeActive().connect( [this] { update_log ( "App became active." ); } );
    getSignalWillResignActive().connect( [this] { update_log ( "App will resign active." ); } );
    
    getWindow()->getSignalDisplayChange().connect( std::bind( &VisibleApp::displayChange, this ) );
    
    // Create an invisible folder for storage
    mUserStorageDirPath = getHomeDirectory()/".Visible";
    if (!fs::exists( mUserStorageDirPath)) fs::create_directories(mUserStorageDirPath);
}

void VisibleApp::QuitApp(){
    quit();
}


void VisibleApp::DrawGUI(){
    // Draw the menu bar
    {
        ui::ScopedMainMenuBar menuBar;
        
        if( ui::BeginMenu( "File" ) ){
            if(ui::MenuItem("Fullscreen")){
                setFullScreen(!isFullScreen());
            }
            if(ui::MenuItem("Is IdLab LIF ", "I")){
                m_isIdLabLif = ! m_isIdLabLif;
            }
            ImGui::Checkbox("Is IdLab LIF", &m_isIdLabLif);
            
            ui::MenuItem("Help", nullptr, &showHelp);
            if(ui::MenuItem("Quit", "ESC")){
                QuitApp();
            }
            ui::EndMenu();
        }
        
        if( ui::BeginMenu( "View" ) ){
            ui::MenuItem( "General Settings", nullptr, &showGUI );
            ui::MenuItem( "Log", nullptr, &showLog );
            ui::EndMenu();
        }
        
        //ui::SameLine(ui::GetWindowWidth() - 60); ui::Text("%4.0f FPS", ui::GetIO().Framerate);
        ui::SameLine(ui::GetWindowWidth() - 60); ui::Text("%4.1f FPS", getAverageFps());
    }
    
    //Draw general settings window
    if(showGUI)
    {
        ui::Begin("General Settings", &showGUI, ImGuiWindowFlags_AlwaysAutoResize);
        

        ui::End();
    }
    
    //Draw the log if desired
    if(showLog){
        app_log.Draw("Log", &showLog);
    }
    
    if(showHelp) ui::OpenPopup("Help");
    //ui::ScopedWindow window( "Help", ImGuiWindowFlags_AlwaysAutoResize );
    std::string buildN =  boost::any_cast<const string&>(mPlist.find("CFBundleVersion")->second);
    buildN = "Visible ( build: " + buildN + " ) ";
    if(ui::BeginPopupModal("Help", &showHelp)){
        ui::TextColored(ImVec4(0.92f, 0.18f, 0.29f, 1.00f), "%s", buildN.c_str());
        ui::Text("Arman Garakani, Darisa LLC");
        if(ui::Button("Copy")) ui::LogToClipboard();
        ui::SameLine();
        //  ui::Text("github.com/");
        ui::LogFinish();
        ui::Text("");
        ui::Text("Mouse over any"); ShowHelpMarker("We did it!"); ui::SameLine(); ui::Text("to show help.");
        ui::Text("Ctrl+Click any slider to set its value manually.");
        ui::EndPopup();
    }
    
}


void VisibleApp::windowMove()
{
    //  update_log("window pos: " + toString(getWindow()->getPos()));
}

void VisibleApp::initData( const fs::path &path )
{
    if(! exists(path))
    {
        std::string msg = path.string() + " is not a valid path to a file ";
        vlogger::instance().console()->info(msg);
    }
    mCurrentLifFilePath = path;
    mLifRef = lif_browser::create(path.string());
    
    auto series = mLifRef->get_all_series ();
    
    for( vector<lif_serie_data>::const_iterator serieIt = series.begin(); serieIt != series.end(); ++serieIt )
        createItem( *serieIt, int(serieIt - series.begin()) );
    
    //@todo For now Assume IDLab Format,
    m_isIdLabLif = true;
    
    // create if needed cache directories for this lif file
    bool ok = VisibleAppControl::make_result_cache_directory_for_lif(path, mLifRef);
    std::string msg = (ok) ? "Cache directories existed or were created" : "Cache directories could not be created";
    vlogger::instance().console()->info(msg);

    
    // state
    if(! mItems.empty()){
        mMouseOverItem        = mItems.end();
        mNewMouseOverItem    = mItems.end();
        mSelectedItem        = mItems.begin();
        mSelectedItem->select( timeline(), mLeftBorder );
    }
    
}

void VisibleApp::createItem( const lif_serie_data &serie, int index )
{
    string title				= serie.name();
    string desc					= " Channels: " + to_string(serie.channelCount()) +
    "   width: " + to_string(serie.dimensions()[0]) +
    "   height: " + to_string(serie.dimensions()[1]);
    
    auto platform = ci::app::Platform::get();
    auto palette_path = platform->getResourceDirectory();
    string paletteFilename        = "palette.png";
    palette_path = palette_path / paletteFilename;
    Surface palette = Surface( loadImage(  palette_path ) ) ;
    gl::Texture2dRef image = gl::Texture::create(Surface( ImageSourceRef( new ImageSourceCvMat( serie.poster() ))));
    vec2 pos( mLeftBorder, mTopBorder + index * mItemSpacing );
    Item item = Item(index, pos, title, desc, palette );
    mItems.push_back( item );
    mImages.push_back( image );
    
}

void VisibleApp::mouseMove( MouseEvent event )
{
    
    mNewMouseOverItem = mItems.end();
    
    for( vector<Item>::iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
        if( itemIt->isPointIn( event.getPos() ) && !itemIt->getSelected() ) {
            mNewMouseOverItem = itemIt;
            
            break;
        }
    }
    
    if( mNewMouseOverItem == mItems.end() ){
        if( mMouseOverItem != mItems.end() && mMouseOverItem != mSelectedItem ){
            mMouseOverItem->mouseOff( timeline() );
            mMouseOverItem = mItems.end();
        }
    } else {
        
        if( mNewMouseOverItem != mMouseOverItem && !mNewMouseOverItem->getSelected() ){
            if( mMouseOverItem != mItems.end() && mMouseOverItem != mSelectedItem )
                mMouseOverItem->mouseOff( timeline() );
            mMouseOverItem = mNewMouseOverItem;
            mMouseOverItem->mouseOver( timeline() );
        }
    }
    
    if( mMouseOverItem != mItems.end() && mMouseOverItem != mSelectedItem ){
        mFgImage = mImages[mMouseOverItem->mIndex];
        //      mFgAlpha = 0.0f;
        //     timeline().apply( &mFgAlpha, 1.0f, 0.4f, EaseInQuad() );
    }
}

void VisibleApp::mouseDown( MouseEvent event )
{
    if( mMouseOverItem != mItems.end() ){
        vector<Item>::iterator prevSelected = mSelectedItem;
        mSelectedItem = mMouseOverItem;
        
        // deselect previous selected item
        if( prevSelected != mItems.end() && prevSelected != mMouseOverItem ){
            prevSelected->deselect( timeline() );
            mBgImage = mFgImage;
            mBgColor = Color::white();
            timeline().apply( &mBgColor, Color::black(), 0.4f, EaseInQuad() );
        }
        
        // select current mouseover item
        mSelectedItem->select( timeline(), mLeftBorder );
        mFgImage = mImages[mSelectedItem->mIndex];
        mFgAlpha = 0.0f;
        timeline().apply( &mFgAlpha, 1.0f, 0.4f, EaseInQuad() );
        mMouseOverItem = mItems.end();
        mNewMouseOverItem = mItems.end();
        
        // Open a LIF Context for this serie
        dispatch_lif_viewer(mSelectedItem->mIndex);
    }
}

void VisibleApp::update()
{
    
    for( vector<Item>::iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
        itemIt->update();
    }
    
}

void VisibleApp::draw()
{
    
    // Note: this function is called once per frame for EACH WINDOW
    gl::ScopedViewport scpViewport( ivec2( 0 ), getWindowSize() );
    
    
    // clear out the window with black
    //    gl::clear( Color( 0, 0, 0 ) );
    if( mVisibleScope ){
        gl::draw( mVisibleScope, getWindowBounds() );
    }
    
    if (! mItems.empty()){
        
        gl::enableAlphaBlending();
        
        gl::setMatricesWindowPersp( getWindowSize() );
        
        // draw background image
        if( mBgImage ){
            gl::color( mBgColor );
            gl::draw( mBgImage, getWindowBounds() );
        }
        
        // draw foreground image
        if( mFgImage ){
            Rectf bounds (mFgImage->getBounds());
            bounds = bounds.getCenteredFit(getWindowBounds(), false);
            gl::color( ColorA( 1.0f, 1.0f, 1.0f, mFgAlpha ) );
            gl::draw( mFgImage, bounds );
        }
        
        // draw swatches
        gl::context()->pushTextureBinding( mSwatchLargeTex->getTarget(), mSwatchLargeTex->getId(), 0 );
        if( mSelectedItem != mItems.end() )
            mSelectedItem->drawSwatches();
        
        gl::context()->bindTexture( mSwatchLargeTex->getTarget(), mSwatchLargeTex->getId(), 0 );
        for( vector<Item>::const_iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
            //  if( ! itemIt->getSelected() )
            itemIt->drawSwatches();
        }
        gl::context()->popTextureBinding( mSwatchLargeTex->getTarget(), 0 );
        
        // turn off textures and draw bgBar
        for( vector<Item>::const_iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
            itemIt->drawBgBar();
        }
        
        // turn on textures and draw text
        gl::color( Color( 1.0f, 1.0f, 1.0f ) );
        for( vector<Item>::const_iterator itemIt = mItems.begin(); itemIt != mItems.end(); ++itemIt ) {
            itemIt->drawText();
        }
    }
    else{
        auto ctr = getWindowCenter();
        ctr.x /= 2;
        gl::draw(mDropFileTextTexture, ctr);
    }
    
    DrawGUI();
    
    
}


void VisibleApp::update_log (const std::string& msg)
{
    if (msg.length() > 2)
        vlogger::instance().console()->info(msg);
}

void VisibleApp::displayChange()
{
    //   update_log ( "window display changed: " + toString(getWindow()->getDisplay()->getBounds()));
    //   console() << "ContentScale = " << getWindowContentScale() << endl;
    //   console() << "getWindowCenter() = " << getWindowCenter() << endl;
}


void VisibleApp::windowClose()
{
    WindowRef win = getWindow();
}



void VisibleApp::mouseDrag( MouseEvent event )
{
    
    cinder::app::App::mouseDrag(event);
}




void VisibleApp::mouseUp( MouseEvent event )
{
    
    cinder::app::App::mouseUp(event);
}


void VisibleApp::keyDown( KeyEvent event )
{
    
    if( event.getChar() == 'f' ) {
        // Toggle full screen when the user presses the 'f' key.
        setFullScreen( ! isFullScreen() );
    }
    else if( event.getCode() == KeyEvent::KEY_ESCAPE ) {
        // Exit full screen, or quit the application, when the user presses the ESC key.
        if( isFullScreen() )
            setFullScreen( false );
        else
            quit();
    }
    
}



void VisibleApp::resize ()
{
    
    
}

CINDER_APP( VisibleApp, RendererGl( RendererGl::Options().msaa( 4 ) ), []( App::Settings *settings ) {
    settings->setWindowSize( APP_WIDTH, APP_HEIGHT );
} )


#pragma GCC diagnostic pop
