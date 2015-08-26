/*
 Copyright (c) 2014, The Cinder Project

 This code is intended to be used with the Cinder C++ library, http://libcinder.org

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and
	the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
	the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/

#include "cinder/audio2/msw/FileMediaFoundation.h"
#include "cinder/audio2/dsp/Converter.h"
#include "cinder/audio2/Debug.h"

#include <mfidl.h>
#include <mfapi.h>
#include <mfreadwrite.h>
#include <propvarutil.h>

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "mfreadwrite.lib")

// TODO: try to minimize the number of copies between IMFSourceReader::ReadSample's IMFSample and loading audio2::Buffer
// - currently uses an intermediate vector<float>
// - want to have minimal copies when loading a buffer for BufferPlayerNode, so
// - this is on hold until audio2::Converter is re-addressed

using namespace std;
using namespace ci;

namespace cinder { namespace audio2 { namespace msw {

bool MediaFoundationInitializer::sIsMfInitialized = false;

namespace {

inline double	nanoSecondsToSeconds( LONGLONG ns )		{ return (double)ns / 10000000.0; } 
inline LONGLONG secondsToNanoSeconds( double seconds )	{ return (LONGLONG)seconds * 10000000; }

} // anonymous namespace

// ----------------------------------------------------------------------------------------------------
// MARK: - SourceFileMediaFoundation
// ----------------------------------------------------------------------------------------------------

SourceFileMediaFoundation::SourceFileMediaFoundation()
	: SourceFile(), mCanSeek( false ), mSeconds( 0 ), mReadBufferPos( 0 ), mFramesRemainingInReadBuffer( 0 )
{
}

SourceFileMediaFoundation::SourceFileMediaFoundation( const DataSourceRef &dataSource )
	: SourceFile(), mDataSource( dataSource ), mCanSeek( false ), mSeconds( 0 ), mReadBufferPos( 0 ), mFramesRemainingInReadBuffer( 0 )
{
	MediaFoundationInitializer::initMediaFoundation();
	initReader();
}

SourceFileRef SourceFileMediaFoundation::clone() const
{
	shared_ptr<SourceFileMediaFoundation> result( new SourceFileMediaFoundation );
	result->mDataSource = mDataSource;
	result->initReader();

	return result;
}

SourceFileMediaFoundation::~SourceFileMediaFoundation()
{
}

size_t SourceFileMediaFoundation::performRead( Buffer *buffer, size_t bufferFrameOffset, size_t numFramesNeeded )
{
	CI_ASSERT( buffer->getNumFrames() >= bufferFrameOffset + numFramesNeeded );

	size_t readCount = 0;
	while( readCount < numFramesNeeded ) {

		// first drain any frames that were previously read from an IMFSample
		if( mFramesRemainingInReadBuffer ) {
			size_t remainingToDrain = min( mFramesRemainingInReadBuffer, numFramesNeeded );

			// TODO: use Buffer::copyChannel
			for( size_t ch = 0; ch < mNativeNumChannels; ch++ ) {
				float *readChannel = mReadBuffer.getChannel( ch ) + mReadBufferPos;
				float *resultChannel = buffer->getChannel( ch );
				memcpy( resultChannel + readCount, readChannel, remainingToDrain * sizeof( float ) );
			}

			mReadBufferPos += remainingToDrain;
			mFramesRemainingInReadBuffer -= remainingToDrain;
			readCount += remainingToDrain;
			continue;
		}

		CI_ASSERT( ! mFramesRemainingInReadBuffer );

		mReadBufferPos = 0;
		size_t outNumFrames = processNextReadSample();
		if( ! outNumFrames )
			break;

		// if the IMFSample num frames is over the specified buffer size, 
		// record how many samples are left over and use up what was asked for.
		if( outNumFrames + readCount > numFramesNeeded ) {
			mFramesRemainingInReadBuffer = outNumFrames + readCount - numFramesNeeded;
			outNumFrames = numFramesNeeded - readCount;
		}

		size_t offset = bufferFrameOffset + readCount;
		for( size_t ch = 0; ch < mNativeNumChannels; ch++ ) {
			float *readChannel = mReadBuffer.getChannel( ch );
			float *resultChannel = buffer->getChannel( ch );
			memcpy( resultChannel + readCount, readChannel, outNumFrames * sizeof( float ) );
		}

		mReadBufferPos += outNumFrames;
		readCount += outNumFrames;
	}

	return readCount;
}

void SourceFileMediaFoundation::performSeek( size_t readPositionFrames )
{
	if( ! mCanSeek ) {
		CI_LOG_E( "cannot seek." );
		return;
	}

	mReadBufferPos = mFramesRemainingInReadBuffer = 0;

	double positionSeconds = (double)readPositionFrames / (double)mSampleRate;
	if( positionSeconds > mSeconds ) {
		CI_LOG_E( "cannot seek beyond end of file (" << positionSeconds << "s)." );
		return;
	}

	LONGLONG position = secondsToNanoSeconds( positionSeconds );
	PROPVARIANT seekVar;
	HRESULT hr = ::InitPropVariantFromInt64( position, &seekVar );
	CI_ASSERT( hr == S_OK );
	hr = mSourceReader->SetCurrentPosition( GUID_NULL, seekVar );
	CI_ASSERT( hr == S_OK );
	hr = PropVariantClear( &seekVar );
	CI_ASSERT( hr == S_OK );
}

// TODO: test setting MF_LOW_LATENCY attribute
void SourceFileMediaFoundation::initReader()
{
	CI_ASSERT( mDataSource );
	mFramesRemainingInReadBuffer = 0;

	::IMFAttributes *attributes;
	HRESULT hr = ::MFCreateAttributes( &attributes, 1 );
	CI_ASSERT( hr == S_OK );
	auto attributesPtr = makeComUnique( attributes );

	::IMFSourceReader *sourceReader;

	if( mDataSource->isFilePath() ) {
		hr = ::MFCreateSourceReaderFromURL( mDataSource->getFilePath().wstring().c_str(), attributesPtr.get(), &sourceReader );
		CI_ASSERT( hr == S_OK );
	}
	else {
		mComIStream = makeComUnique( new ComIStream( mDataSource->createStream() ) );
		::IMFByteStream *byteStream;
		hr = ::MFCreateMFByteStreamOnStream( mComIStream.get(), &byteStream );
		CI_ASSERT( hr == S_OK );
		mByteStream = makeComUnique( byteStream );

		hr = ::MFCreateSourceReaderFromByteStream( byteStream, attributesPtr.get(), &sourceReader );
		CI_ASSERT( hr == S_OK );
	}

	mSourceReader = makeComUnique( sourceReader );

	// get files native format
	::IMFMediaType *nativeType; // FIXME: i think this is leaked after creation
	::WAVEFORMATEX *fileFormat;
	UINT32 formatSize;
	hr = mSourceReader->GetNativeMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, &nativeType );
	CI_ASSERT( hr == S_OK );
	hr = ::MFCreateWaveFormatExFromMFMediaType( nativeType, &fileFormat, &formatSize );
	CI_ASSERT( hr == S_OK );

	GUID outputSubType = MFAudioFormat_PCM; // default to PCM, upgrade if we can.
	mSampleType = SampleType::INT_16;
	mBytesPerSample = 2;

	if( fileFormat->wBitsPerSample == 32 ) {
		mSampleType = SampleType::FLOAT_32;
		mBytesPerSample = 4;
		outputSubType = MFAudioFormat_Float;
	}

	mNumChannels = mNativeNumChannels = fileFormat->nChannels;
	mSampleRate = mNativeSampleRate = fileFormat->nSamplesPerSec;

	::CoTaskMemFree( fileFormat );

	mReadBuffer.setSize( mMaxFramesPerRead, mNativeNumChannels );

	// set output type, which loads the proper decoder:
	::IMFMediaType *outputType;
	hr = ::MFCreateMediaType( &outputType );
	auto outputTypeRef = makeComUnique( outputType );
	hr = outputTypeRef->SetGUID( MF_MT_MAJOR_TYPE, MFMediaType_Audio );
	CI_ASSERT( hr == S_OK );
	hr = outputTypeRef->SetGUID( MF_MT_SUBTYPE, outputSubType );
	CI_ASSERT( hr == S_OK );

	hr = mSourceReader->SetCurrentMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, outputTypeRef.get() );
	CI_ASSERT( hr == S_OK );

	// after the decoder is loaded, we have to now get the 'complete' output type before retrieving its format
	// TODO: still needed?
	// - format seems to always have a reliable bits per sample
	::IMFMediaType *completeOutputType;
	hr = mSourceReader->GetCurrentMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, &completeOutputType );
	CI_ASSERT( hr == S_OK );

	::WAVEFORMATEX *format;
	hr = ::MFCreateWaveFormatExFromMFMediaType( completeOutputType, &format, &formatSize );
	CI_ASSERT( hr == S_OK );
	::CoTaskMemFree( format );

	// get seconds:
	::PROPVARIANT durationProp;
	hr = mSourceReader->GetPresentationAttribute( MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &durationProp );
	CI_ASSERT( hr == S_OK );
	LONGLONG duration = durationProp.uhVal.QuadPart;
	
	mSeconds = nanoSecondsToSeconds( duration );
	mNumFrames = mFileNumFrames = size_t( mSeconds * (double)mSampleRate );

	::PROPVARIANT seekProp;
	hr = mSourceReader->GetPresentationAttribute( MF_SOURCE_READER_MEDIASOURCE, MF_SOURCE_READER_MEDIASOURCE_CHARACTERISTICS, &seekProp );
	CI_ASSERT( hr == S_OK );
	ULONG flags = seekProp.ulVal;
	mCanSeek = ( ( flags & MFMEDIASOURCE_CAN_SEEK ) == MFMEDIASOURCE_CAN_SEEK );
}

size_t SourceFileMediaFoundation::processNextReadSample()
{
	::IMFSample *mediaSample;
	DWORD streamFlags = 0;
	LONGLONG timeStamp;
	HRESULT hr = mSourceReader->ReadSample( MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, NULL, &streamFlags, &timeStamp, &mediaSample );
	CI_ASSERT( hr == S_OK );

	if( streamFlags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED ) {
		CI_LOG_E( "type change unhandled" );
		return 0;
	}
	if( streamFlags & MF_SOURCE_READERF_ENDOFSTREAM ) {
		// end of file
		return 0;
	}
	if( ! mediaSample ) {
		// out of samples
		mediaSample->Release();
		return 0;
	}

	auto samplePtr = makeComUnique( mediaSample );

	DWORD bufferCount;
	hr = samplePtr->GetBufferCount( &bufferCount );
	CI_ASSERT( hr == S_OK );

	CI_ASSERT( bufferCount == 1 ); // just looking out for a file type with more than one buffer.. haven't seen one yet.

	// get the buffer
	::IMFMediaBuffer *mediaBuffer;
	BYTE *audioData = NULL;
	DWORD audioDataLength;

	hr = samplePtr->ConvertToContiguousBuffer( &mediaBuffer );
	hr = mediaBuffer->Lock( &audioData, NULL, &audioDataLength );

	size_t numChannels = mNativeNumChannels;
	size_t numFramesRead = audioDataLength / ( mBytesPerSample * numChannels );

	mReadBuffer.setNumFrames( numFramesRead );

	if( mSampleType == SampleType::FLOAT_32 ) {
		float *sourceFloatSamples = (float *)audioData;
		if( numChannels == 1)
			memcpy( mReadBuffer.getData(), sourceFloatSamples, numFramesRead * sizeof( float ) );
		else
			dsp::deinterleave( sourceFloatSamples, mReadBuffer.getData(), mReadBuffer.getNumFrames(), numChannels, numFramesRead );
	}
	else if( mSampleType == SampleType::INT_16 ) {
		int16_t *sourceInt16Samples = (int16_t *)audioData;
		dsp::deinterleave( sourceInt16Samples, mReadBuffer.getData(), mReadBuffer.getNumFrames(), numChannels, numFramesRead );
	}
	else
		CI_ASSERT_NOT_REACHABLE();

	hr = mediaBuffer->Unlock();
	CI_ASSERT( hr == S_OK );

	mediaBuffer->Release();
	return numFramesRead;
}

// ----------------------------------------------------------------------------------------------------
// MARK: - TargetFileMediaFoundation
// ----------------------------------------------------------------------------------------------------

namespace {

::GUID getMfAudioFormat( SampleType sampleType )
{
	switch( sampleType ) {
		case SampleType::INT_16:	return ::MFAudioFormat_PCM;
		case SampleType::FLOAT_32:	return ::MFAudioFormat_Float;
		default: break;
	}

	CI_ASSERT_NOT_REACHABLE();
	return MFAudioFormat_PCM;
}

size_t getSampleSize( SampleType sampleType )
{
	switch( sampleType ) {
		case SampleType::INT_16:	return 2;
		case SampleType::FLOAT_32:	return 4;
		default: break;
	}

	CI_ASSERT_NOT_REACHABLE();
	return 0;
}

} // anonymous namespace

TargetFileMediaFoundation::TargetFileMediaFoundation( const DataTargetRef &dataTarget, size_t sampleRate, size_t numChannels, SampleType sampleType, const std::string &extension )
	: TargetFile( dataTarget, sampleRate, numChannels, sampleType ), mStreamIndex( 0 )
{
	MediaFoundationInitializer::initMediaFoundation();

	::IMFSinkWriter *sinkWriter;
	HRESULT hr = ::MFCreateSinkWriterFromURL( dataTarget->getFilePath().wstring().c_str(), NULL, NULL, &sinkWriter );
	CI_ASSERT( hr == S_OK );

	mSinkWriter = makeComUnique( sinkWriter );

	mSampleSize = getSampleSize( mSampleType );
	const UINT32 bitsPerSample = 8 * mSampleSize;
	const WORD blockAlignment = mNumChannels * mSampleSize;
	const DWORD averageBytesPerSecond = mSampleRate * blockAlignment;

	// Set the output media type.

	IMFMediaType *mediaType;
	hr = ::MFCreateMediaType( &mediaType );
	CI_ASSERT( hr == S_OK );
	auto mediaTypeManaged = makeComUnique( mediaType );

	hr = mediaType->SetGUID( MF_MT_MAJOR_TYPE, MFMediaType_Audio );
	CI_ASSERT( hr == S_OK );

	const ::GUID audioFormat = getMfAudioFormat( mSampleType );
	hr = mediaType->SetGUID( MF_MT_SUBTYPE, audioFormat );
	CI_ASSERT( hr == S_OK );

	hr = mediaType->SetUINT32( MF_MT_AUDIO_SAMPLES_PER_SECOND, (UINT32)mSampleRate );
	CI_ASSERT( hr == S_OK );

	hr = mediaType->SetUINT32( MF_MT_AUDIO_BITS_PER_SAMPLE, bitsPerSample );
	CI_ASSERT( hr == S_OK );

	hr = mediaType->SetUINT32( MF_MT_AUDIO_BLOCK_ALIGNMENT, blockAlignment );
	CI_ASSERT( hr == S_OK );

	hr = mediaType->SetUINT32( MF_MT_AUDIO_AVG_BYTES_PER_SECOND, averageBytesPerSecond );
	CI_ASSERT( hr == S_OK );

	hr = mediaType->SetUINT32( MF_MT_AUDIO_NUM_CHANNELS, (UINT32)mNumChannels );
	CI_ASSERT( hr == S_OK );

	hr = mediaType->SetUINT32( MF_MT_ALL_SAMPLES_INDEPENDENT, 1 );
	CI_ASSERT( hr == S_OK );

	hr = mediaType->SetUINT32( MF_MT_FIXED_SIZE_SAMPLES, 1 );
	CI_ASSERT( hr == S_OK );

	hr = mSinkWriter->AddStream( mediaType, &mStreamIndex );
	CI_ASSERT( hr == S_OK );

	CI_LOG_V( "initializing complete. filePath: " << dataTarget->getFilePath() << ", stream index: " << mStreamIndex );

	// Tell the sink writer to start accepting data.
	hr = mSinkWriter->BeginWriting();
	CI_ASSERT( hr == S_OK );
}

TargetFileMediaFoundation::~TargetFileMediaFoundation()
{
	if( mSinkWriter ) {
		HRESULT hr = mSinkWriter->Finalize();
		CI_ASSERT( hr == S_OK );
	}
}

void TargetFileMediaFoundation::performWrite( const Buffer *buffer, size_t numFrames, size_t frameOffset )
{
	// create media sample
	::IMFSample *mediaSample;
	HRESULT hr = ::MFCreateSample( &mediaSample );
	CI_ASSERT( hr == S_OK );

	auto samplePtr = makeComUnique( mediaSample );

	double lengthSeconds = (double)numFrames / (double)mSampleRate;
	const LONGLONG sampleDuration = secondsToNanoSeconds( lengthSeconds );
	hr = mediaSample->SetSampleDuration( sampleDuration );
	CI_ASSERT( hr == S_OK );

	double currentTimeSeconds = (double)frameOffset / (double)mSampleRate;
	const LONGLONG sampleTime = secondsToNanoSeconds( currentTimeSeconds );
	hr = mediaSample->SetSampleTime( sampleTime );
	CI_ASSERT( hr == S_OK );

	// create media buffer and fill with audio samples.

	DWORD bufferSizeBytes = numFrames * buffer->getNumChannels() * mSampleSize;
	::IMFMediaBuffer *mediaBuffer;
	hr = ::MFCreateMemoryBuffer( bufferSizeBytes, &mediaBuffer );
	CI_ASSERT( hr == S_OK );

	hr = mediaBuffer->SetCurrentLength( bufferSizeBytes );
	CI_ASSERT( hr == S_OK );

	hr = mediaSample->AddBuffer( mediaBuffer );
	CI_ASSERT( hr == S_OK );

	BYTE *audioData;
	hr = mediaBuffer->Lock( &audioData, NULL, NULL );
	CI_ASSERT( hr == S_OK );

	if( mSampleType == SampleType::FLOAT_32 ) {
		float *destFloatSamples = (float *)audioData;
		if( mNumChannels == 1 )
			memcpy( destFloatSamples, buffer->getData(), numFrames * mSampleSize );
		else
			dsp::interleave( buffer->getData(), destFloatSamples, buffer->getNumFrames(), mNumChannels, numFrames );
	}
	else if( mSampleType == SampleType::INT_16 ) {
		int16_t *destInt16Samples = (int16_t *)audioData;
		dsp::interleave( buffer->getData(), destInt16Samples, buffer->getNumFrames(), mNumChannels, numFrames );
	}
	else
		CI_ASSERT_NOT_REACHABLE();

	hr = mediaBuffer->Unlock();
	CI_ASSERT( hr == S_OK );

	hr = mSinkWriter->WriteSample( mStreamIndex, mediaSample );
	CI_ASSERT( hr == S_OK );
}

// ----------------------------------------------------------------------------------------------------
// MARK: - MediaFoundationImpl (Startup / Shutdown)
// ----------------------------------------------------------------------------------------------------

// static
void MediaFoundationInitializer::initMediaFoundation()
{
	if( ! sIsMfInitialized ) {
		sIsMfInitialized = true;
		HRESULT hr = ::MFStartup( MF_VERSION );
		CI_ASSERT( hr == S_OK );
	}
}

// static
void MediaFoundationInitializer::shutdownMediaFoundation()
{
	if( sIsMfInitialized ) {
		sIsMfInitialized = false;
		HRESULT hr = ::MFShutdown();
		CI_ASSERT( hr == S_OK );
	}
}

} } } // namespace cinder::audio2::msw
