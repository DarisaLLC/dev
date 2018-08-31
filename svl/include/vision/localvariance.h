//
//  localvariance.h
//
//  Created by Arman Garakani on 1/3/14.
//  Copyright (c) 2014 Arman Garakani. All rights reserved.
//

#ifndef irec_framework_localvariance_h
#define irec_framework_localvariance_h

#include "opencv2/core/core.hpp"


namespace svl
{

    /*!
     Local Variance using Integral Image
     */
    class localVAR
    {
    public:
        CV_WRAP explicit localVAR( cv::Size filter_size );

        bool process (const cv::Mat& image, cv::Mat& results ) const;
            //! Produces local variance image for this image using kernel size
            // if filter size is less than 2 in either side, returns false
        

        const cv::Mat& VarianceImageRef () const { return m_var; }
        float min_variance () const { return m_minVar; }
        float max_variance () const { return m_maxVar; }
        void reset () const ;
        void clear_cache () const ;

    protected:
        mutable cv::Size m_fsize;
        mutable cv::Mat m_s, m_ss, m_var;
        mutable cv::Mat m_single; // single channel;
        mutable cv::Mat m_mask; // Original image
        mutable float m_minVar, m_maxVar;
        mutable cv::Size m_isize;

    private:
        bool use_cached_images (const cv::Mat&) const;
        void allocate_images (const cv::Mat&) const;
        void convert_or_not (const cv::Mat&) const;

    };
    
}


#endif
