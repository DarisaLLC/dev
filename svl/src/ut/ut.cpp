
#include <iostream>
#include "gtest/gtest.h"
#include <memory>
#include "boost/filesystem.hpp"
#include "vision/histo.h"
#include "vision/drawUtils.hpp"
#include "vision/gradient.h"
#include "vision/ipUtils.h"
#include "vision/edgel.h"
#include "core/pair.hpp"
#include "vision/roiWindow.h"
#include "vision/rowfunc.h"
#include "vision/gauss.hpp"
#include "vision/gmorph.hpp"
#include "vision/sample.hpp"
#include "core/stl_utils.hpp"
#include "vision/labelconnect.hpp"
#include "vision/registration.h"
#include "cinder_cv/cinder_xchg.hpp"
#include "ut_localvar.hpp"
#include "core/cv_gabor.hpp"
#include "opencv2/highgui.hpp"
#include "ut_similarity.hpp"
#include "otherIO/lifFile.hpp"
#include "core/gtest_env_utils.hpp"
#include "vision/histo.h"
#include "vision/roiMultiWindow.h"
#include "vision/sample.hpp"
#include "vision/opencv_utils.hpp"


using namespace svl;


//#define _INTERACTIVE
void on_mouse( int e, int x, int y, int d, void *ptr )
{
    cv::Point*p = (cv::Point*)ptr;
    p->x = x;
    p->y = y;
    cout<<*p;
}

int iAngle = 180;
int iScale = 50;
int iBorderMode = 0;
Mat showImage ;
int iImageCenterY = 0;
int iImageCenterX = 0;
const char* displayImage = "Display Image";

void CallbackForTrackBar(int, void*)
{
    iImageCenterY = showImage.rows / 2.0;
    iImageCenterX = showImage.cols / 2.0;
    Mat matRotation = getRotationMatrix2D(  cv::Point( iImageCenterX, iImageCenterY ), (iAngle - 180), iScale / 50.0 );
    
    // Rotate the image
    Mat imgRotated;
    warpAffine( showImage, imgRotated, matRotation, showImage.size(), INTER_LINEAR, iBorderMode, Scalar() );
    imshow( displayImage, imgRotated );
    
    
}


static test_utils::genv * dgenv_ptr;

std::string  expectedNames []
{
    "Series003"
    , "Series007"
    , "Series010"
    , "Series013"
    , "Series016"
    , "Series019"
    , "Series022"
    , "Series026"
    , "Series029"
    , "Series032"
    , "Series035"
    , "Series038"
    , "Series041"
    , "Series045"
};

std::string  expected3Names []
{
    "Series006"
    , "Series007"
    , "Series009"
    , "Series012"
    , "Series014"
    , "Series018"
    , "Series023"
    , "Series024"
    , "Series027"
    , "Series029"
    , "Series035"
    , "Series037"
    , "Preview038"
};


const char * pi341[] =
{
    "0000000000000000",
    "0000000000000000",
    "0000000000000000",
    "0000000000000000",
    "0000011110000000",
    "0000001111000000",
    "0000000111100000",
    "0000000011110000",
    "0000000000000000"
    "0000000000000000",
    "0000000000000000",
    "0000000000000000", 0};

const char * pi341234[] =
{
    "0000000000000000",
    "0000000000000000",
    "0000000000000000",
    "0000000000000000",
    "0000012340000000",
    "0000001234000000",
    "0000000123400000",
    "0000000012340000",
    "0000000000000000"
    "0000000000000000",
    "0000000000000000",
    "0000000000000000", 0};


std::shared_ptr<uint16_t> create_ramp16(int asp, int pct = 100)
{
    static int full_pct(100);
    int count = 65536; // 256 * 256
    int cliff = (count * pct) / full_pct;
    int w = 256;
    int h = 256;
    int index = 0;
    std::shared_ptr<uint16_t> image = test_utils::create<uint16_t>(w / asp, h * asp);
    uint16_t * pptr = image.get();
    
    for (; index < count; index++)
    {
        pptr[index] = (index < cliff) ? index : cliff;
    }
    
    return image;
}




std::shared_ptr<uint16_t> create_easy_ramp16(int w, int h)
{
    int count = w * h;
    int index = 0;
    std::shared_ptr<uint16_t> image = test_utils::create<uint16_t>(w, h);
    uint16_t * pptr = image.get();
    
    for (; index < count; index++)
    {
        pptr[index] = (uint16_t)std::sqrt(index);
    }
    
    return image;
}



TEST (ut_lifFile, single_channel)
{
    std::string filename ("Sample1.lif");
    std::pair<test_utils::genv::path_t, bool> res = dgenv_ptr->asset_path(filename);
    EXPECT_TRUE(res.second);
    lifIO::LifReader lif(res.first.string());
    cout << "LIF version "<<lif.getVersion() << endl;
    EXPECT_EQ(14, lif.getNbSeries() );
    size_t serie = 0;
    size_t frame = 0;
    lifIO::LifSerie& se0 = lif.getSerie(serie);
    const std::vector<size_t>& dims = se0.getSpatialDimensions();
    EXPECT_EQ(512, dims[0]);
    EXPECT_EQ(128, dims[1]);
    std::vector<std::string> series;
    for (auto ss = 0; ss < lif.getNbSeries(); ss++)
        series.push_back (lif.getSerie(ss).getName());
    
    EXPECT_EQ(sizeof(expectedNames)/sizeof(std::string), series.size ());
    
    for (auto sn = 0; sn < series.size(); sn++)
    {
        EXPECT_EQ(0, expectedNames[sn].compare(series[sn]));
        EXPECT_EQ(250, lif.getSerie(sn).getNbTimeSteps());
        EXPECT_EQ(65536, lif.getSerie(sn).getNbPixelsInOneTimeStep ());
        EXPECT_EQ(1, lif.getSerie(sn).getChannels().size());
    }
    
    
    roiWindow<P8U> slice (dims[0], dims[1]);
    lif.getSerie(0).fill2DBuffer(slice.rowPointer(0));
    histoStats h;
    h.from_image(slice);
    EXPECT_NEAR(h.mean(), 114.271, 0.001);
    EXPECT_NEAR(h.median(), 114.0, 0.001);
    EXPECT_NEAR(h.min(), 44.0, 0.001);
    EXPECT_NEAR(h.max(), 255.0, 0.001);
    
    
    
}

TEST (ut_lifFile, triple_channel)
{
    std::string filename ("3channels.lif");
    std::pair<test_utils::genv::path_t, bool> res = dgenv_ptr->asset_path(filename);
    EXPECT_TRUE(res.second);
    lifIO::LifReader lif(res.first.string());
    cout << "LIF version "<<lif.getVersion() << endl;
    EXPECT_EQ(13, lif.getNbSeries() );
    size_t serie = 0;
    size_t frame = 0;
    lifIO::LifSerie& se0 = lif.getSerie(serie);
    const std::vector<size_t>& dims = se0.getSpatialDimensions();
    EXPECT_EQ(512, dims[0]);
    EXPECT_EQ(128, dims[1]);
    std::vector<std::string> series;
    for (auto ss = 0; ss < lif.getNbSeries(); ss++)
        series.push_back (lif.getSerie(ss).getName());
    
    EXPECT_EQ(sizeof(expected3Names)/sizeof(std::string), series.size ());
    
    {
        EXPECT_EQ(0, expected3Names[0].compare(series[0]));
        EXPECT_EQ(31, lif.getSerie(0).getNbTimeSteps());
        EXPECT_EQ(65536, lif.getSerie(0).getNbPixelsInOneTimeStep ());
        EXPECT_EQ(3, lif.getSerie(0).getChannels().size());
    }
    for (auto sn = 1; sn < (series.size()-1); sn++)
    {
        EXPECT_EQ(0, expected3Names[sn].compare(series[sn]));
        EXPECT_EQ(500, lif.getSerie(sn).getNbTimeSteps());
        EXPECT_EQ(65536, lif.getSerie(sn).getNbPixelsInOneTimeStep ());
        EXPECT_EQ(3, lif.getSerie(sn).getChannels().size());
    }
    {
        EXPECT_EQ(0, expected3Names[series.size()-1].compare(series[series.size()-1]));
        EXPECT_EQ(1, lif.getSerie(series.size()-1).getNbTimeSteps());
        EXPECT_EQ(262144, lif.getSerie(series.size()-1).getNbPixelsInOneTimeStep ());
        EXPECT_EQ(3, lif.getSerie(series.size()-1).getChannels().size());
    }
    
    roiWindow<P8U> slice (dims[0], dims[1]);
    lif.getSerie(0).fill2DBuffer(slice.rowPointer(0));
    histoStats h;
    h.from_image(slice);
    EXPECT_NEAR(h.mean(), 5.82, 0.001);
    EXPECT_NEAR(h.median(), 0.0, 0.001);
    EXPECT_NEAR(h.min(), 0.0, 0.001);
    EXPECT_NEAR(h.max(), 205.0, 0.001);
    

}

TEST(basicU8, gradient)
{
    
    std::string filename ("small_read_ut.txt");
    std::pair<test_utils::genv::path_t, bool> res = dgenv_ptr->asset_path(filename);
    roiWindow<P8U> pels (res.first.string());
    
    auto w = pels.width();
    auto h = pels.height();
    roiWindow<P8U> mag(w, h);
    roiWindow<P8U> ang(w, h);
    roiWindow<P8U> peaks(w, h);
    Gradient(pels, mag, ang);
    
    unsigned int pks = SpatialEdge(mag, ang, peaks, 1, false);
    
    EXPECT_EQ(pks, 26);
    
    
    std::string afilename ("small_mag_ut.txt");
    std::pair<test_utils::genv::path_t, bool> ares = dgenv_ptr->asset_path(afilename);
    roiWindow<P8U> au_mag (ares.first.string());
    
    EXPECT_EQ(mag.size(), au_mag.size());
    for (auto j = 0; j < mag.height(); j++)
        for (auto i = 0; i < mag.width(); i++)
        {
            EXPECT_EQ(mag.getPixel(i,j), au_mag.getPixel(i,j));
        }
    
    std::string mfilename ("small_ang_ut.txt");
    std::pair<test_utils::genv::path_t, bool> mres = dgenv_ptr->asset_path(mfilename);
    roiWindow<P8U> au_ang (mres.first.string());
    
    EXPECT_EQ(ang.size(), au_ang.size());
    for (auto j = 0; j < ang.height(); j++)
        for (auto i = 0; i < ang.width(); i++)
        {
            EXPECT_EQ(ang.getPixel(i,j), au_ang.getPixel(i,j));
        }
    
#ifdef NOTYET
    std::string pfilename ("small_peak_ut.txt");
    std::pair<test_utils::genv::path_t, bool> pres = dgenv_ptr->asset_path(pfilename);
    roiWindow<P8U> au_peaks (pres.first.string());
    au_peaks.output();
    
    EXPECT_EQ(peaks.size(), au_peaks.size());
    for (auto j = 0; j < peaks.height(); j++)
        for (auto i = 0; i < peaks.width(); i++)
        {
            EXPECT_EQ(peaks.getPixel(i,j), au_peaks.getPixel(i,j));
        }
#endif
    
    
    
    
}

TEST(timing8, gradient)
{
    std::shared_ptr<uint8_t> img1 = test_utils::create_trig(1920, 1080);
    double endtime;
    std::clock_t start;
    
    int w = 1920, h = 1080;
    roiWindow<P8U> pels(w, h);
    roiWindow<P8U> mag(w, h);
    roiWindow<P8U> ang(w, h);
    roiWindow<P8U> peaks(w, h);
    
    start = std::clock();
    Gradient(pels, mag, ang);
    unsigned int pks = SpatialEdge(mag, ang, peaks, 1, false);
    endtime = (std::clock() - start) / ((double)CLOCKS_PER_SEC);
    double scale = 1000.0;
    std::cout << " Edge: Mag, Ang, NonMax: 1920 * 1080 * 8 bit " << endtime * scale << " millieseconds per " << std::endl;
    
    std::vector<svl::feature> features;
    auto pks_2 = SpatialEdge(mag, ang, features, 1);
    EXPECT_EQ(pks, pks_2);
    
}



TEST(basicU8, motioncenter)
{
    roiWindow<P8U> tmp(16, 16);
    roiWindow<P8U> mag(16, 16);
    roiWindow<P8U> ang(16, 16);
    roiWindow<P8U> peaks(16, 16);
    
    tmp.set(255);
    iPair tl(tmp.width() / 4, tmp.height() / 4);
    iPair br(4 * tmp.width() / 4 - 1, 4 * tmp.height() / 4 - 1);
    roiWindow<P8U> box(tmp, tl.x(), tl.y(), br.x() - tl.x(), br.y() - tl.y());
    box.set(0xef);
    
    Gradient(tmp, mag, ang);
    
    unsigned int pks = SpatialEdge(mag, ang, peaks, 1, false);
    EXPECT_EQ(pks, 37);
    fPair center;
    GetMotionCenter(peaks, ang, center);
    fPair correct(3.52f, 3.52f);
    EXPECT_EQ(true, fabs(center.first - correct.first) < 0.1f);
    EXPECT_EQ(true, fabs(center.second - correct.second) < 0.1f);
  
    
}

void fillramp (roiWindow<P8U>& img)
{
    
    for (auto index = 0; index < img.n(); index++)
    {
        img.setPixel(index / img.width(), index % img.height(), (uint8_t)(index % 256));
    }
    
}
TEST (ut_mi, basic)
{
    // cv::namedWindow( "Image View", 1 );
    //    NewFromSVL (img3, mv);
    //    imshow( "Image View", mv );
    //    key = cvWaitKey( -1);
    
    //    Gauss3by3(img1, img3);
    
    
    std::vector<std::string> names = {"meena_mp67_mp67.png", "meena.png", "meena_pp67_pp67.png", "meena_p1p3_p1p3.png"};
    std::vector<roiWindow<P8U> > images;
    std::vector<std::pair<test_utils::genv::path_t, bool> > results;
    std::vector<MutualInfo::Parts8U> outputs;
    
    for (auto name : names)
    {
        results.push_back(dgenv_ptr->asset_path(name));
    }
    
    for (auto res : results)
    {
        if (res.second)
        {
            std::pair<Surface8uRef, Channel8uRef> wp = svl::image_io_read_surface(res.first);
            images.push_back (svl::NewFromChannel(*wp.second, 0));
            //   cv::Mat mv;
            //   NewFromSVL (images.back(), mv);
            //   imshow( "Image View", mv );
            //   int key = cvWaitKey( -1);
            
        }
    }
    
    if (results[1].second)
    {
        MutualInfo::Parts8U outp;
        MutualInfo::getMI(images[1], images[1], outp);
        std::cout << outp;
    }
    
    if (results[1].second && results[0].second)
    {
        MutualInfo::Parts8U outp;
        MutualInfo::getMI(images[1], images[0], outp);
        std::cout << outp;
    }
    
    if (results[1].second && results[2].second)
    {
        MutualInfo::Parts8U outp;
        MutualInfo::getMI(images[1], images[2], outp);
        std::cout << outp;
    }
    
    if (results[1].second && results[3].second)
    {
        MutualInfo::Parts8U outp;
        MutualInfo::getMI(images[1], images[3], outp);
        std::cout << outp;
    }
    
}



TEST (ut_similarity, run)
{
    UT_similarity tester;
    tester.run();
}

TEST(basic, self_registration)
{
    
}
TEST(basic, image_txt_io)
{
    std::string filename ("small_read_ut.txt");
    std::pair<test_utils::genv::path_t, bool> res = dgenv_ptr->asset_path(filename);
    roiWindow<P8U> r8 (res.first.string());
    EXPECT_EQ(r8.width(), 20);
    EXPECT_EQ(r8.height(), 11);
}

TEST(basicU8, png_io)
{
    std::string name = "checkerboard_u8.png";
    std::pair<test_utils::genv::path_t, bool> res = dgenv_ptr->asset_path(name);
    EXPECT_TRUE(res.second);
    {
        std::pair<Surface8uRef, Channel8uRef> wp = svl::image_io_read_surface(res.first);
        roiWindow<P8U> rootwin = svl::NewFromChannel(*wp.second, 0);
        EXPECT_EQ(rootwin.width(), 512);
        EXPECT_EQ(rootwin.height(), 512);
    }
}




TEST(basicU8, jpg_io)
{
    {
        std::string name = "check_u8.jpg";
        std::pair<test_utils::genv::path_t, bool> res = dgenv_ptr->asset_path(name);
        EXPECT_TRUE(res.second);
        {
            std::pair<Surface8uRef, Channel8uRef> wp = svl::image_io_read_surface(res.first);
            roiWindow<P8U> rootwin = svl::NewFromChannel(*wp.second, 0);
            EXPECT_EQ(rootwin.width(), 962);
            EXPECT_EQ(rootwin.height(), 602);
        }
    }
    
    {
        std::string name = "check_rgb8.jpg";
        std::pair<test_utils::genv::path_t, bool> res = dgenv_ptr->asset_path(name);
        EXPECT_TRUE(res.second);
        {
#ifdef NEEDS_UPDATE
            std::pair<Surface8uRef, Channel8uRef> wp (svl::image_io_read_surface(res.first));
            roiWindow<P8UC4> rootwin = svl::NewFromSurface(wp.first.get());
            EXPECT_EQ(rootwin.width(), 962);
            EXPECT_EQ(rootwin.height(), 602);
#endif
        }
    }
}


TEST(basicU8, localvar)
{
    ut_localvar ut;
    ut.test_method ();
    
}


TEST(synth, basic)
{
    roiWindow<P8U> tmp(17, 17);
    tmp.set();
    tmp.setPixel(8, 8, 255);
    tmp.setPixel(4, 4, 255);
    tmp.setPixel(12, 12, 255);
    tmp.setPixel(4, 12, 255);
    tmp.setPixel(12, 4, 255);
    roiWindow<P8U> smooth(tmp.width(), tmp.height());
    Gauss3by3(tmp, smooth);
    
    //    smooth.print_pixel();
}

TEST(basicip, timing)
{
    std::string name = "checkerboard_u8.png";
    std::pair<test_utils::genv::path_t, bool> res = dgenv_ptr->asset_path(name);
    if (res.second)
    {
        std::pair<Surface8uRef, Channel8uRef> wp = svl::image_io_read_surface(res.first);
        roiWindow<P8U> rootwin = svl::NewFromChannel(*wp.second, 0);
        int width = rootwin.width();
        int height = rootwin.height();
        
        roiWindow<P8U> smooth(width, height);
        
        {
            std::clock_t start;
            start = std::clock();
            Gauss3by3(rootwin, smooth);
            double endtime = (std::clock() - start) / ((double)CLOCKS_PER_SEC);
            std::cout << " 3x3 Gaussian + 1x3 gauss on the outer 1 pixel " << endtime * 1000 << " millieseconds per " << std::endl;
        }
    }
}




TEST(basicmorph, area)
{
    roiWindow<P8U> pels(17, 25);
    roiWindow<P8U> dst (17, 25);
    
    pels.set(255.0);
    pels.setPixel(8, 12, 13);
    
    PixelMin3by3 (pels, dst);
    
    EXPECT_EQ(dst.getPixel(7, 11), 13);
    
    pels.set(1.0);
    pels.setPixel(8, 12, 13);
    
    PixelMax3by3 (pels, dst);
    
    EXPECT_EQ(dst.getPixel(7, 11), 13);
    
}

TEST(basicgauss, area)
{
    roiWindow<P8U> pels(17, 25);
    roiWindow<P8U> dst (17, 25);
    
    pels.set();
    pels.setPixel(8, 12, 255);
    
    Gauss3by3 (pels, dst);
    
    EXPECT_EQ(dst.getPixel(8, 12), 64);
    
    //pels.print_pixel();
    //dst.print_pixel();
}

TEST(basicgauss, edge)
{
    roiWindow<P8U> pels(17, 25);
    roiWindow<P8U> dst (17, 25);
    
    pels.set();
    pels.setPixel(16, 24, 255);
    
    Gauss3by3 (pels, dst);
    
    EXPECT_EQ(dst.getPixel(16, 24), 255);
    EXPECT_EQ(dst.getPixel(15, 23), 16);
    EXPECT_EQ(dst.getPixel(15, 24), 48);
    EXPECT_EQ(dst.getPixel(16, 23), 48);
    
    pels.set();
    pels.setPixel(0, 0, 255);
    
    Gauss3by3 (pels, dst);
    
    EXPECT_EQ(dst.getPixel(0, 0), 255);
    EXPECT_EQ(dst.getPixel(1, 1), 16);
    EXPECT_EQ(dst.getPixel(0, 1), 48);
    EXPECT_EQ(dst.getPixel(1, 0), 48);
    
    pels.set();
    pels.setPixel(0, 24, 255);
    
    Gauss3by3 (pels, dst);
    
    EXPECT_EQ(dst.getPixel(0, 24), 255);
    EXPECT_EQ(dst.getPixel(1, 23), 16);
    EXPECT_EQ(dst.getPixel(0, 23), 48);
    EXPECT_EQ(dst.getPixel(1, 24), 48);
    
    
    pels.set();
    pels.setPixel(16, 0, 255);
    
    Gauss3by3 (pels, dst);
    
    EXPECT_EQ(dst.getPixel(16, 0), 255);
    EXPECT_EQ(dst.getPixel(15, 1), 16);
    EXPECT_EQ(dst.getPixel(15, 0), 48);
    EXPECT_EQ(dst.getPixel(16, 1), 48);
    
    //    pels.print_pixel();
    //    dst.print_pixel();
}

TEST(basicU8, labelConnect)
{
    const char * frame[] =
    {
        "00100100",
        "00110100",
        "00011000",
        "01000100",
        "01000000",
        0};
    const char * gold[] =
    {
        "00100100",
        "00110100",
        "00011000",
        "02000100",
        "02000000",
        0};
    
    roiWindow<P8U> pels(8,5);
    DrawShape(pels, frame);
    
    roiWindow<P8U> label(8,5);
    DrawShape(label, gold);
    
    labelConnect<P8U> lc (pels);
    lc.run();
    EXPECT_EQ(label.size(), lc.label().size());
    for (auto j = 0; j < label.height(); j++)
        for (auto i = 0; i < label.width(); i++)
        {
            EXPECT_EQ(label.getPixel(i,j), lc.label().getPixel(i,j));
        }
    
    
    simpleRLE<P8U> srlep8(label);
    {
        std::vector<PixelType<P8U>::run_t> rles;
        int rle_count = srlep8.rle_row ((uint8_t*) label.rowPointer(0), label.width(), rles);
        EXPECT_EQ(rle_count, 5);
        EXPECT_EQ(rles[0].first, 0);
        EXPECT_EQ(rles[0].second, 2);
        EXPECT_EQ(rles[1].first, 1);
        EXPECT_EQ(rles[1].second, 1);
        EXPECT_EQ(rles[2].first, 0);
        EXPECT_EQ(rles[2].second, 2);
        EXPECT_EQ(rles[3].first, 1);
        EXPECT_EQ(rles[3].second, 1);
        EXPECT_EQ(rles[4].first, 0);
        EXPECT_EQ(rles[4].second, 2);
    }
    
    
    
}

TEST(basicU8, histo)
{
    
    const char * frame[] =
       {
        "00000000000000000000",
        "00001020300000000000",
        "00000000000000000000",
        "00000000012300000000",
        "00000000000000000000",
        0};
    
    roiWindow<P8U> pels(20, 5);
    DrawShape(pels, frame);
    
    histoStats hh;
    hh.from_image<P8U>(pels);
    EXPECT_EQ(hh.max(0), 3);
    EXPECT_EQ(hh.min(0), 0);
}


TEST(basicU8, hyst)
{
    const char * frame[] =
    {
        "00000000000000000000",
        "00001020300000000000",
        "00000000000000000000",
        "00000112000000000000",
        "00000000300000000000",
        "00000000000000000000",
        "00000000000000000000",
        0};
    
    roiWindow<P8U> fpels(22, 7);
    DrawShape(fpels, frame);
    roiWindow<P8U> dst;
    int32_t got = -1;
    hystheresisThreshold::U8(fpels, dst, 1, 3, got, 0);
#ifdef _INTERACTIVE
    dst.output();
#endif
    EXPECT_EQ(5, got);
}





TEST(basic, corr8)
{
    std::shared_ptr<uint8_t> img1 = test_utils::create_trig(1920, 1080);
    std::shared_ptr<uint8_t> img2 = test_utils::create_trig(1920, 1080);
    
    basicCorrRowFunc<uint8_t> corrfunc(img1.get(), img2.get(), 1920, 1920, 1920, 1080);
    corrfunc.areaFunc();
    CorrelationParts cp;
    corrfunc.epilog(cp);
    EXPECT_EQ(cp.r(), 1);
}


TEST(basic, corr16)
{
    std::shared_ptr<uint16_t> img1 = create_easy_ramp16(1920, 1080);
    std::shared_ptr<uint16_t> img2 = create_easy_ramp16(1920, 1080);
    
    basicCorrRowFunc<uint16_t> corrfunc(img1.get(), img2.get(), 1920, 1920, 1920, 1080);
    corrfunc.areaFunc();
    CorrelationParts cp;
    corrfunc.epilog(cp);
    EXPECT_EQ(cp.r(), 1);
}


TEST(timing8, corr)
{
    std::shared_ptr<uint8_t> img1 = test_utils::create_trig(1920, 1080);
    std::shared_ptr<uint8_t> img2 = test_utils::create_trig(1920, 1080);
    double endtime;
    std::clock_t start;
    int l;
    start = std::clock();
    int num_loops = 1000;
    // Time setting and resetting
    for (l = 0; l < num_loops; ++l)
    {
        basicCorrRowFunc<uint8_t> corrfunc(img1.get(), img2.get(), 1920, 1920, 1920, 1080);
        corrfunc.areaFunc();
        CorrelationParts cp;
        corrfunc.epilog(cp);
    }
    endtime = (std::clock() - start) / ((double)CLOCKS_PER_SEC);
    double scale = 1000.0 / (num_loops);
    std::cout << " Correlation: 1920 * 1080 * 8 bit " << endtime * scale << " millieseconds per " << std::endl;
}

TEST(timing16, corr)
{
    std::shared_ptr<uint16_t> img1 = create_easy_ramp16(1920, 1080);
    std::shared_ptr<uint16_t> img2 = create_easy_ramp16(1920, 1080);
    double endtime;
    std::clock_t start;
    int l;
    start = std::clock();
    int num_loops = 1000;
    // Time setting and resetting
    for (l = 0; l < num_loops; ++l)
    {
        basicCorrRowFunc<uint16_t> corrfunc(img1.get(), img2.get(), 1920, 1920, 1920, 1080);
        corrfunc.areaFunc();
        CorrelationParts cp;
        corrfunc.epilog(cp);
    }
    endtime = (std::clock() - start) / ((double)CLOCKS_PER_SEC);
    double scale = 1000.0 / (num_loops);
    std::cout << " Correlation: 1920 * 1080 * 16 bit " << endtime * scale << " millieseconds per " << std::endl;
}



/***
 **  Use --gtest-filter=*NULL*.*shared" to select https://code.google.com/p/googletest/wiki/V1_6_AdvancedGuide#Running_Test_Programs%3a_Advanced_Options
 */


int main(int argc, char ** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: --assets <fully qualified path> to directory containing data \n";
        exit(1);
    }
    
    boost::filesystem::path installpath = std::string(argv[0]);
    
    std::string current_exec_name = argv[0]; // Name of the current exec program
    std::vector<std::string> all_args;
    all_args.assign(argv + 1, argv + argc);
    
    boost::filesystem::path input;
    bool got_input;
    got_input = false;
    std::string input_marker("--assets");
    
    for (unsigned ac = 0; ac < all_args.size(); ac++)
    {
        if (!got_input)
        {
            std::size_t found = all_args.at(ac).find(input_marker);
            
            if (found != std::string::npos && ac < (all_args.size() - 1))
            {
                input = all_args[ac + 1];
                got_input = true;
            }
        }
    }
    
    std::cout << input.string() << std::endl;
    if (!boost::filesystem::exists(input))
    {
        std::cerr << input.string() << " does not exist. Use --help\n";
        exit(1);
    }
    
    boost::filesystem::path assetpath = input;
    // Gtest will take over the allocated testing environment.
    dgenv_ptr = new test_utils::genv(installpath.string());
    ::testing::Environment * const DSgEnv = ::testing::AddGlobalTestEnvironment(dgenv_ptr);
    dgenv_ptr->add_content_directory(assetpath);
    
    testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
#ifdef _INTERACTIVE
    while (true)
        ;
#endif
}

