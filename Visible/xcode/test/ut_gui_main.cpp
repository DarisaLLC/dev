//
//  ut_gui_main.cpp
//  Visible
//
//  Created by Arman Garakani on 1/3/19.
//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-private-field"
#pragma GCC diagnostic ignored "-Wcomma"

#include "VisibleApp.h"
#include "imguivariouscontrols.h"
#include "imgui_internal.h"
#include <algorithm>

std::vector<double> acid = {39.1747, 39.2197, 39.126, 39.0549, 39.0818, 39.0655, 39.0342,
    38.8791, 38.8527, 39.0099, 38.8608, 38.9188, 38.8499, 38.6693,
    38.2513, 37.9095, 37.3313, 36.765, 36.3621, 35.7261, 35.0656,
    34.2602, 33.2523, 32.3183, 31.6464, 31.0073, 29.8166, 29.3423,
    28.5223, 27.5152, 26.8191, 26.3114, 25.8164, 25.0818, 24.7631,
    24.6277, 24.8184, 25.443, 26.2479, 27.8759, 29.2094, 30.7956,
    32.3586, 33.6268, 35.1586, 35.9315, 36.808, 37.3002, 37.67, 37.9986,
    38.2788, 38.465, 38.5513, 38.6818, 38.8076, 38.9388, 38.9592,
    39.058, 39.1322, 39.0803, 39.1779, 39.1531, 39.1375, 39.1978,
    39.0379, 39.1231, 39.202, 39.1581, 39.1777, 39.2971, 39.2366,
    39.1555, 39.2822, 39.243, 39.1807, 39.1488, 39.2491, 39.265, 39.198,
    39.2855, 39.2595, 39.4274, 39.3258, 39.3162, 39.4143, 39.3034,
    39.2099, 39.2775, 39.5042, 39.1446, 39.188, 39.2006, 39.1799,
    39.4077, 39.2694, 39.1967, 39.2828, 39.2438, 39.2093, 39.2167,
    39.2749, 39.4703, 39.2846};


#include "Resources.h"

using namespace std;

#ifndef IMGUI_DEFINE_MATH_OPERATORS

//static ImVec2 operator+(const ImVec2 &a, const ImVec2 &b) {
//    return ImVec2(a.x + b.x, a.y + b.y);
//}


//static ImVec2 operator*(const ImVec2 &a, const ImVec2 &b) {
//    return ImVec2(a.x * b.x, a.y * b.y);
//}
//
//static ImVec2 operator/(const ImVec2 &a, const ImVec2 &b) {
//    return ImVec2(a.x / b.x, a.y / b.y);
//}
//
//static ImVec2 operator*(const ImVec2 &a, const float b) {
//    return ImVec2(a.x * b, a.y * b);
//}
#endif

#define TEST_APP_WIDTH 1280
#define TEST_APP_HEIGHT 960
bool gbIsPlaying = false;
bool gPlayLoop = false;

void SetStyle()
{
    ImGuiStyle &st = ImGui::GetStyle();
    st.FrameBorderSize = 1.0f;
    st.FramePadding = ImVec2(4.0f,2.0f);
    st.ItemSpacing = ImVec2(8.0f,2.0f);
    st.WindowBorderSize = 1.0f;
    //   st.TabBorderSize = 1.0f;
    st.WindowRounding = 1.0f;
    st.ChildRounding = 1.0f;
    st.FrameRounding = 1.0f;
    st.ScrollbarRounding = 1.0f;
    st.GrabRounding = 1.0f;
    //  st.TabRounding = 1.0f;
    
    // Setup style
    ImVec4* colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 0.95f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.13f, 0.12f, 0.12f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.00f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.94f);
    colors[ImGuiCol_Border] = ImVec4(0.53f, 0.53f, 0.53f, 0.46f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.85f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.22f, 0.22f, 0.40f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.16f, 0.16f, 0.53f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.48f, 0.48f, 0.48f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.79f, 0.79f, 0.79f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.48f, 0.47f, 0.47f, 0.91f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.56f, 0.55f, 0.55f, 0.62f);
    colors[ImGuiCol_Button] = ImVec4(0.50f, 0.50f, 0.50f, 0.63f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.67f, 0.67f, 0.68f, 0.63f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.26f, 0.26f, 0.26f, 0.63f);
    colors[ImGuiCol_Header] = ImVec4(0.54f, 0.54f, 0.54f, 0.58f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.64f, 0.65f, 0.65f, 0.80f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.25f, 0.25f, 0.25f, 0.80f);
    colors[ImGuiCol_Separator] = ImVec4(0.58f, 0.58f, 0.58f, 0.50f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.81f, 0.81f, 0.81f, 0.64f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.81f, 0.81f, 0.81f, 0.64f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.87f, 0.87f, 0.87f, 0.53f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.87f, 0.87f, 0.87f, 0.74f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.87f, 0.87f, 0.87f, 0.74f);
    //   colors[ImGuiCol_Tab] = ImVec4(0.01f, 0.01f, 0.01f, 0.86f);
    //   colors[ImGuiCol_TabHovered] = ImVec4(0.29f, 0.29f, 0.79f, 1.00f);
    //   colors[ImGuiCol_TabActive] = ImVec4(0.31f, 0.31f, 0.91f, 1.00f);
    //   colors[ImGuiCol_TabUnfocused] = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
    //   colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.19f, 0.19f, 0.19f, 1.00f);
    //  colors[ImGuiCol_DockingPreview] = ImVec4(0.38f, 0.48f, 0.60f, 1.00f);
    //  colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.68f, 0.68f, 0.68f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.77f, 0.33f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.87f, 0.55f, 0.08f, 1.00f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.47f, 0.60f, 0.76f, 0.47f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(0.58f, 0.58f, 0.58f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
}

//MySequence mySequence(gNodeDelegate);


class VisibleProtoApp : public App, public gui_base
{
public:
    
    //   VisibleRunApp();
    //  ~VisibleRunApp();
    
    
    virtual void QuitApp(){
        ImGui::DestroyContext();
        // fg::ThreadsShouldStop = true;
        quit();
    }
    
    void prepareSettings( Settings *settings );
    
    void setup()override{
#if 1
        ui::initialize(ui::Options()
                       .window(getWindow())
                       .itemSpacing(vec2(6, 6)) //Spacing between widgets/lines
                       .itemInnerSpacing(vec2(10, 4)) //Spacing between elements of a composed widget
                       .color(ImGuiCol_Button, ImVec4(0.86f, 0.93f, 0.89f, 0.39f)) //Darken the close button
                       .color(ImGuiCol_Border, ImVec4(0.86f, 0.93f, 0.89f, 0.39f))
                       //  .color(ImGuiCol_TooltipBg, ImVec4(0.27f, 0.57f, 0.63f, 0.95f))
                       );
#endif
        
        SetStyle ();
        setFrameRate( 60 );
        setWindowPos(getWindowSize()/4);
        getWindow()->setAlwaysOnTop();
        
        
        ntrack.first = " Length ";
       timedVecOfVals_t& data = ntrack.second;
        data.resize(acid.size());
        for (int tt = 0; tt < acid.size(); tt++){
            data[tt].second = acid[tt];
            data[tt].first.first = tt;
            data[tt].first.second = time_spec_t(tt / 1000.0);
        }
        m_tracks_ref = std::make_shared<vecOfNamedTrack_t> ();
        m_tracks_ref->push_back(ntrack);
        
        duration_time_t duration;
        duration.first = ntrack.second.front().first;
        duration.second = ntrack.second.back().first;
        mySequence.mFrameMin = 0;
        mySequence.mFrameMax = (int) acid.size() - 1;
        mySequence.mSequencerItemTypeNames = {" PCI "};
        mySequence.editable_plot_data.load(m_tracks_ref->at(0), 0);
        
    }
    
    void mouseDown( MouseEvent event )override{
        cinder::app::App::mouseDown(event);
    }
    void keyDown( KeyEvent event )override{
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
        else if(event.getCode() == 'd' ) {
            
        }
    }
    virtual void SetupGUIVariables() override{
        //Set up global preferences for the GUI
        //http://anttweakbar.sourceforge.net/doc/tools:anttweakbar:twbarparamsyntax
        GetGUI().DefineGlobal("iconalign=horizontal");
        mLoop = gl::Texture::create( loadImage( loadResource( IMAGE_PLOOP )));
        mNoLoop = gl::Texture::create( loadImage( loadResource( IMAGE_PNLOOP  )));
        
    }
    virtual void DrawGUI() override{
        ImGuiIO& io = ImGui::GetIO();
        //    ImGui::ScopedMainMenuBar menubar;
        ui::SameLine(ui::GetWindowWidth() - 60); ui::Text("%4.1f FPS", getAverageFps());
        
        ImVec2 rc = ImGui::GetItemRectSize();
        ImVec2 deltaHeight = ImVec2(0, rc.y);
        ImGui::SetNextWindowPos(deltaHeight);
        auto tmp = io.DisplaySize - deltaHeight;
        ImGui::SetNextWindowSize(tmp);
        //  int mSelectedNodeIndex = 0;
        int64_t firstFrame = mFrameMin;
        int64_t currentFrame = 0;
        
        if (ImGui::Begin("Visible", 0, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |ImGuiWindowFlags_NoScrollWithMouse |
                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBringToFrontOnFocus))
        {
            if (ImGui::Begin("Timeline"))
            {
                int selectedEntry =  0; //gNodeDelegate.mSelectedNodeIndex;
                
                ImGui::PushItemWidth(80);
                ImGui::PushID(200);
                ImGui::InputInt("", &mFrameMin, 0, 0);
                ImGui::PopID();
                ImGui::SameLine();
                if (ImGui::Button("|<"))
                    currentFrame = mFrameMin;
                ImGui::SameLine();
                if (ImGui::Button("<"))
                    currentFrame--;
                
                ImGui::SameLine();
                if (ImGui::Button(">"))
                    currentFrame++;
                ImGui::SameLine();
                if (ImGui::Button(">|"))
                    currentFrame = mFrameMax;
                ImGui::SameLine();
                extern bool gbIsPlaying;
                if (ImGui::Button(gbIsPlaying ? "Stop" : "Play"))
                {
                    if (!gbIsPlaying)
                    {
                        currentFrame = mFrameMin;
                    }
                    gbIsPlaying = !gbIsPlaying;
                }

                extern bool gPlayLoop;
                if(mNoLoop == nullptr){
                    mNoLoop = gl::Texture::create( loadImage( loadResource( IMAGE_PNLOOP  )));
                }
                if (mLoop == nullptr){
                    mLoop = gl::Texture::create( loadImage( loadResource( IMAGE_PLOOP )));
                }
                
                unsigned int playNoLoopTextureId = mNoLoop->getId();//  evaluation.GetTexture("Stock/PlayNoLoop.png");
                unsigned int playLoopTextureId = mLoop->getId(); //evaluation.GetTexture("Stock/PlayLoop.png");
                
                ImGui::SameLine();
                if (ImGui::ImageButton((ImTextureID)(uint64_t)(gPlayLoop ? playLoopTextureId : playNoLoopTextureId), ImVec2(16.f, 16.f)))
                    gPlayLoop = !gPlayLoop;
                
               
           
            
                ImSequencer::Sequencer(&mySequence, &currentFrame, NULL, &selectedEntry, &firstFrame, ImSequencer::SEQUENCER_EDIT_ALL |  ImSequencer::SEQUENCER_ADD |  ImSequencer::SEQUENCER_DEL);
                std::cout << "T: " << currentFrame << "  E: " << selectedEntry << "  FF: " << firstFrame << std::endl;
            }
            ImGui::End();

            // let's create the sequencer
            static int selectedEntry = -1;
            static int64 firstFrame = 0;
            static bool expanded = true;
            
            ImGui::SetNextWindowPos(ImVec2(300, 300));
            ImGui::SetNextWindowSize(ImVec2(512, 480));
            ImGui::Begin(" Results ");
            
            //    ImGui::PushItemWidth(130);
            //    ImGui::InputInt("Frame Min", &mySequence.mFrameMin);
            //    ImGui::SameLine();
            //    ImGui::InputInt("Frame ", &currentFrame);
            //    ImGui::SameLine();
            //    ImGui::InputInt("Frame Max", &mySequence.mFrameMax);
            //    ImGui::PopItemWidth();
            
            Sequencer(&mySequence, &currentFrame, &expanded, &selectedEntry, &firstFrame, ImSequencer::SEQUENCER_EDIT_NONE );
            // add a UI to edit that particular item
            if (selectedEntry != -1)
            {
                const timeLineSequence::timeline_item &item = mySequence.myItems[selectedEntry];
                ImGui::Text("I am a %s, please edit me", mySequence.mSequencerItemTypeNames[item.mType].c_str());
                // switch (type) ....
            }
            
            ImGui::End();
            
            
            
#if 0
            if (ImGui::Begin("Nodes"))
            {
                ImGui::PushItemWidth(150);
                //    ImGui::InputText("Name", &material.mName);
                ImGui::SameLine();
                ImGui::PopItemWidth();
                ImGui::PushItemWidth(60);
                //  static int previewSize = 0;
                //ImGui::Combo("Preview size", &previewSize, "  128\0  256\0  512\0 1024\0 2048\0 4096\0");
                ImGui::SameLine();
                if (ImGui::Button("Do exports"))
                {
                    //         nodeGraphDelegate.DoForce();
                }
                ImGui::SameLine();
                if (ImGui::Button("Save Graph"))
                {
                }
                ImGui::SameLine();
                if (ImGui::Button("Delete Graph"))
                {
                    //     library.mMaterials.erase(library.mMaterials.begin() + selectedMaterial);
                    //   selectedMaterial = int(library.mMaterials.size()) - 1;
                    //    UpdateNewlySelectedGraph(nodeGraphDelegate, evaluation);
                }
                ImGui::PopItemWidth();
            }
            ImGui::End();
            
            if (ImGui::Begin("Shaders")){}
            ImGui::End();
            
            if (ImGui::Begin("Library"))
            {
                if (ImGui::Button("New Graph")){}
                ImGui::SameLine();
                if (ImGui::Button("Import")){}
            }
            ImGui::End();
            
            //   ImGui::SetWindowSize(ImVec2(300, 300));
            if (ImGui::Begin("Parameters")){}
            ImGui::End();
#endif
            
        }
        
        ImGui::End();
    }
    
    
    
    void draw()override{
        gl::clear( Color::gray( 0.5f ) );
        DrawGUI();
    }
    
    bool shouldQuit();
    
private:
    ci::gl::TextureRef    mNoLoop, mLoop;
    namedTrack_t ntrack;
    std::shared_ptr<vecOfNamedTrack_t> m_tracks_ref;
    timeLineSequence mySequence;
    int mFrameMin, mFrameMax;
    
    
};


CINDER_APP( VisibleProtoApp, RendererGl( RendererGl::Options().msaa( 4 ) ), []( App::Settings *settings ) {
    settings->setWindowSize( TEST_APP_WIDTH, TEST_APP_HEIGHT );
} )


