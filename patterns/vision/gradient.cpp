
#include "angle_units.hpp"
#include "gradient.h"
#include "pair.hpp"
#include "rowfunc.h"
#include <assert.h>
#include "boost/math/constants/constants.hpp"

using namespace std;


static EdgeTables sEdgeTables;
static double pi = boost::math::constants::pi<double>();

/*
 *  Does not respect the IplROI
 *  sobel 3x3 the outer row and col are undefined
 */


static void sobelEdgeProcess_b(const roiWindow<P8U> & image, roiWindow<P8U> & magnitudes, roiWindow<P8U> & angles)
{
    // I/O images the same size. First sobel out put is at halfK where kernel is 3x3
    assert(image.width() == magnitudes.width());
    assert(image.width() == angles.width());
    assert(image.height() == magnitudes.height());
    assert(image.height() == angles.height());

    int magUpdate = magnitudes.rowUpdate(); // rowPixelUpdate();
    int angleUpdate = angles.rowUpdate();   // rowPixelUpdate();
    int srcUpdate = image.rowUpdate();      // rowPixelUpdate();
    const uint8_t * src0 = image.pelPointer(0, 0);

    iPair kernel_3(3, 3);
    iPair halfK = kernel_3 / 2;

    uint8_t * mag = magnitudes.pelPointer(halfK.first, halfK.second);
    uint8_t * angle = angles.pelPointer(halfK.first, halfK.second);

    int h(angles.height() - 2);
    int width(angles.width() - 2);

    // For 8 bit images, we divide by 8 (4+4). For 16 bit images, we further divide by 256
    // This could be parameterized as pixel depth is often 12 or 14 bits and not full 16 bits

    const int normBits = 3;

    const uint8_t * magTable = sEdgeTables.magnitudeTable();
    const uint8_t * angTable = sEdgeTables.angleTable();

    // Sobel kernels are symertic. We use a L C R roll over algorithm

    do
    {
        const uint8_t * src = src0;
        uint8_t * m = mag;
        uint8_t * a = angle;

        int left(src[0] + 2 * src[srcUpdate] + src[2 * srcUpdate]);
        int yKernelLeft(-src[0] + src[2 * srcUpdate]);
        src++;
        int center(src[0] + 2 * src[srcUpdate] + src[2 * srcUpdate]);
        int yKernelCenter(-src[0] + src[2 * srcUpdate]);
        src++;
        int w(width);

        do
        {
            int right(src[0] + 2 * src[srcUpdate] + src[2 * srcUpdate]);
            int yKernelRight(-src[0] + src[2 * srcUpdate]);

            // Calculate x and y magnitudes
            int product = yKernelLeft + 2 * yKernelCenter + yKernelRight;
            
            int8_t xmag (right >= left ? ((right - left + 3) >> 3) : ((right - left + 4) >> 3));
            
            int8_t ymag (product >= 0 ? ((product + 3) >> 3) : ((product + 4) >> 3));

            
            int index = (xmag << EdgeTables::eMagPrecision) | std::abs(ymag);

            if (index >= EdgeTables::eMagTableSize)
            {
                std::cerr << index << (int)xmag << "," << (int)ymag << "," << w << "," << h << "," << right << "," << left << endl;
            }
            assert(index < EdgeTables::eMagTableSize);

            *m = magTable[index];
            m++;
            
            uint32_t angleIndex;
            angleIndex = uint8_t(xmag);
            angleIndex <<= 8;
            angleIndex |= uint8_t(ymag);
            *a++ = angTable[angleIndex];
           

            left = center;
            center = right;
            yKernelLeft = yKernelCenter;
            yKernelCenter = yKernelRight;
            src++;
        } while (--w);
        mag += magUpdate;
        angle += angleUpdate;
        src0 += srcUpdate;
    } while (--h);
}

/*
 * TruePeak - This is a 3 pixel operator that selects the middle pixel
 * if it is a peak. This pixel is deemed a peak if either
 * a) it is greater than both its neighbours
 * b) it is greater than a neighbour on the "black" side of the image
 *    and equal to the neighbour on the other side
 *
 * This is fairly inefficient implementation.
 */

static inline uint8_t& angleIndex(uint8_t *start,uint32_t x,uint32_t y)
{
    uint32_t index = (uint8_t)(x) << 8;
    index |= (uint8_t)y;
    return start[index];
}

// Max Magnitude from one of the kernels is ([1 2 1] * 255) / 4 or 255
// We are using 7bits for magnitude and 16 bits for precision for angle.
// @note really wasteful. Have to design later.


void EdgeTables::init()
{
    mMagScale = 255 / 256.;
    int x, y, mapped;
    double realVal;
    mMagTable.resize(eMagTableSize);
    mThetaTable.resize(eAtanRange + 1);

    // Magnitude Table: Linear Scale.
    uint8_t * atIndex = &mMagTable[0];
    for (x = 0; x <= eMaxMag; x++)
        for (y = 0; y <= eMaxMag; y++)
        {
            realVal = sqrt(double(y * y + x * x));
            mapped = (int)(std::min(realVal * mMagScale, 255.0) + 0.5);
            *atIndex++ = uint8_t(mapped);
        }

    const double ascale(256.0 / (2 * pi));
    uint8_t* a = &mThetaTable[0];
    for(x = -127; x <= 127; x++)
        for(y = -127; y <= 127; y++)
        {
            int32_t val;
            double angle(atan2((double)y,(double)x));
            if(angle < 0)
                angle += 2*pi;
            val = ((int32_t)(ascale*angle + 0.5)) % 256;
            angleIndex(a,x,y) = uint8_t(val);
        }
}


inline uint8_t EdgeTables::binAtan(int y, int x) const
{
    static uint8_t piover2(1 << ((sizeof(uint8_t) * 8) - 2)), pi(1 << ((sizeof(uint8_t) * 8) - 1)), m3PIover2(piover2 + pi);
    static int precision(eAtanPrecision);

    const uint8_t * ThetaTable = angleTable();

    if (y < 0)
    {
        y = -y;
        if (x >= 0)
        {
            if (y <= x)
                return -ThetaTable[(y << precision) / x];
            else
                return (m3PIover2 + ThetaTable[(x << precision) / y]);
        }
        else
        {
            x = -x;
            if (y <= x)
                return (pi + ThetaTable[(y << precision) / x]);
            else
                return (m3PIover2 - ThetaTable[(x << precision) / y]);
        }
    }

    else
    {
        if (x >= 0)
        {
            if (x == 0 && y == 0) return 0;
            if (y <= x)
                return ThetaTable[(y << precision) / x];
            else
                return (piover2 - ThetaTable[(x << precision) / y]);
        }
        else
        {
            x = -x;
            if (y <= x)
                return (pi - ThetaTable[(y << precision) / x]);
            else
                return (piover2 + ThetaTable[(x << precision) / y]);
        }
    }
}


// Computes the 8-connected direction from the 8 bit angle
// @todo support 16bit and true peak
static inline int getAxis(uint8_t val)
{
    return ((val + (1 << (8 - 4))) >> (8 - 3)) % 8;
}


/*
 * mag / ang are |pad| ==> peaks are |pad|pad|
 * Sobel is 3x3. Peak Detection is also 3x3 ==> 5x5 operation
 * Peak is same size and translation as mag and ang which are the same as input image
 */


unsigned int SpatialEdge(const roiWindow<P8U> & magImage, const roiWindow<P8U> & angleImage, roiWindow<P8U> & peaks_, uint8_t threshold, bool angleLabeled)
{
    assert(magImage.width() > 0);
    assert(angleImage.height() > 0);
    assert(magImage.width() == peaks_.width());
    assert(magImage.width() == angleImage.width());
    assert(magImage.height() == peaks_.height());
    assert(magImage.height() == angleImage.height());

    unsigned int numPeaks = 0;
    iPair kernel_5(5, 5);
    iPair halfK = kernel_5 / 2;
    int magUpdate(magImage.rowUpdate());
    int dstUpdate(peaks_.rowUpdate());


    for (int y = halfK.y(); y < peaks_.height() - halfK.y(); y++)
    {
        const uint8_t * mag = magImage.pelPointer(halfK.x(), y);
        const uint8_t * ang = angleImage.pelPointer(halfK.x(), y);
        uint8_t * dest = peaks_.pelPointer(halfK.x(), y);

        for (int x = halfK.x(); x < peaks_.width() - halfK.x(); x++, mag++, ang++, dest++)
        {
            if (*mag < threshold)
            {
                *dest = 0;
                continue;
            }

            uint8_t m1, m2;
            int angle = getAxis(*ang);

            switch (angle)
            {
            case 0:
            case 4:
                m1 = *(mag - 1);
                m2 = *(mag + 1);
                break;

            case 1:
            case 5:
                m1 = *(mag - 1 - magUpdate);
                m2 = *(mag + 1 + magUpdate);
                break;

            case 2:
            case 6:
                m1 = *(mag - magUpdate);
                m2 = *(mag + magUpdate);
                break;

            case 3:
            case 7:
                m1 = *(mag + 1 - magUpdate);
                m2 = *(mag - 1 + magUpdate);
                break;

            default:
                m1 = m2 = 0;
                assert(false);
            }

            if ((*mag > m1 && *mag >= m2) || (*mag >= m1 && *mag > m2))
            {
                numPeaks++;
                *dest = (angleLabeled) ? (128 + ((*ang) >> 1)) : *mag;
            }
            else
                *dest = 0;
        }
    }

    peaks_.setBorder(halfK.x());
    return numPeaks;
}


void Gradient(const roiWindow<P8U> & image, roiWindow<P8U> & magnitudes, roiWindow<P8U> & angles)
{
    iPair kernel_3(3, 3);
    iPair halfK = kernel_3 / 2;
    sobelEdgeProcess_b(image, magnitudes, angles);
    // Clear the half kernel at all sides
    magnitudes.setBorder(halfK.x());
    angles.setBorder(halfK.x());
}

bool GetMotionCenter(const roiWindow<P8U> & peaks, const roiWindow<P8U> & ang, fPair & center)
{
    MotionCenter mc;
    int width = peaks.width();
    int height = peaks.height();
    float x = peaks.bound().ul().first;
    float y = peaks.bound().ul().second;

    for (float j = 0; j < height; j++)
    {
        uint8_t * pptr = peaks.rowPointer(j);
        uint8_t * aptr = ang.rowPointer(j);
        for (float i = 0; i < width; i++)
        {
            uint8_t p = *pptr++;
            uint8_t a = (*aptr++ + 64) % 256;
            if (p)
            {
                uAngle8 a8(a);
                fPair uv, pos;
                uv.first = static_cast<float>(sin(a8));
                uv.second = static_cast<float>(cos(a8));
                pos.first = x + i;
                pos.second = y + j;
                mc.add(pos, uv);
            }
        }
    }
    return mc.center(center);
}


void hysteresisThreshold(const roiWindow<P8U> & magImage,
                         roiWindow<P8U> & dst,
                         uint8_t magLowThreshold, uint8_t magHighThreshold,
                         int32_t & nSurvivors, int32_t nPels)
{
    const uint8_t noEdgeLabel = 0;
    const uint8_t edgeLabel = 128;
    uint8_t oscEdgeLabel = 255;
    int width = magImage.width();
    int height = magImage.height();

    if (!dst.isBound())
    {
        roiWindow<P8U> dstroot(width + 2, height + 2);
        dstroot.setBorder(1, noEdgeLabel);
        dst = roiWindow<P8U>(dstroot, 1, 1, width, height);
    }

    dst.set(noEdgeLabel);
    roiWindow<P8U> work(width + 2, height + 2);


    // Create the identity map for this set of thresholds
    vector<uint8_t> map(256);
    int32_t i(0);
    for (; i < magLowThreshold; i++)
        map[i] = noEdgeLabel;
    for (i = 0; i < (magHighThreshold - magLowThreshold); i++)
        map[magLowThreshold + i] = oscEdgeLabel;
    for (i = 0; i < (256 - magHighThreshold); i++)
        map[magHighThreshold + i] = edgeLabel;

    uint8_t * mag = magImage.pelPointer(0, 0);
    uint8_t * wptr = work.pelPointer(1, 1);

    pixelMap<uint8_t, uint8_t> convert(mag, wptr, magImage.rowUpdate(), work.rowUpdate(), width, height, map);
    convert.areaFunc();
    work.setBorder(1, noEdgeLabel);


    const int32_t maxPeaks(nPels <= 0 ? width * height : nPels);

    vector<hystheresisThreshold::hystNode> stack(maxPeaks);

    int32_t nPeaks(0);
    i = 0;
    const int32_t workUpdate(work.rowUpdate());
    const int32_t dstUpdate(dst.rowUpdate());

    // Note:
    // -- and ++ are push and pop operations for the stack
    for (int32_t y = 0; y < height; y++)
    {
        uint8_t * p = work.pelPointer(1, y);
        uint8_t * dest = dst.pelPointer(1, y);

        for (int32_t x = width; x--; p++, dest++)
        {
            if (*p == edgeLabel)
            {
                stack[i].workAddr = p;
                stack[(i++)].destAddr = dest;

                while (i)
                {
                    uint8_t * pp = stack[(--i)].workAddr;
                    uint8_t * dd = stack[i].destAddr;

                    // If it is not marked, mark it an edge in dest
                    // and mark it a no edge in mag
                    if (!*dd)
                    {
                        *dd = edgeLabel;
                        nPeaks++;
                    }
                    *pp = noEdgeLabel;

                    // If any of our neighbors is a possible edge push their address in

                    if (*(pp - workUpdate - 1) == oscEdgeLabel)
                    {
                        stack[i].workAddr = pp - workUpdate - 1;
                        stack[(i++)].destAddr = dd - dstUpdate - 1;
                    }
                    if (*(pp - workUpdate) == oscEdgeLabel)
                    {
                        stack[i].workAddr = pp - workUpdate;
                        stack[(i++)].destAddr = dd - dstUpdate;
                    }
                    if (*(pp - workUpdate + 1) == oscEdgeLabel)
                    {
                        stack[i].workAddr = pp - workUpdate + 1;
                        stack[(i++)].destAddr = dd - dstUpdate + 1;
                    }
                    if (*(pp + 1) == oscEdgeLabel)
                    {
                        stack[i].workAddr = pp + 1;
                        stack[(i++)].destAddr = dd + 1;
                    }
                    if (*(pp + workUpdate + 1) == oscEdgeLabel)
                    {
                        stack[i].workAddr = pp + workUpdate + 1;
                        stack[(i++)].destAddr = dd + dstUpdate + 1;
                    }
                    if (*(pp + workUpdate) == oscEdgeLabel)
                    {
                        stack[i].workAddr = pp + workUpdate;
                        stack[(i++)].destAddr = dd + dstUpdate;
                    }
                    if (*(pp + workUpdate - 1) == oscEdgeLabel)
                    {
                        stack[i].workAddr = pp + workUpdate - 1;
                        stack[(i++)].destAddr = dd + dstUpdate - 1;
                    }
                    if (*(pp - 1) == oscEdgeLabel)
                    {
                        stack[i].workAddr = pp - 1;
                        stack[(i++)].destAddr = dd - 1;
                    }
                }
            }
        }
    }

    nSurvivors = nPeaks;
    return;
}
