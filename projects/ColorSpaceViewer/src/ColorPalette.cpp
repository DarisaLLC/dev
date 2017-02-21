#include "ColorPalette.h"

#include "cinder/CinderAssert.h"
#include "cinder/Rand.h"

#include <algorithm>
#include <numeric>

using namespace ci;
using namespace col;

/////////////////////////////////////////////////////////////////////////////////
///////////////////////////// MEDIAN CUT VEC CUBE ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

RgbCube::RgbCube( const std::vector<ivec3>& colors )
: mColorVecs( colors )
{
	// Here we set the bounds of the cube
	mBounds[0] = { 255, 0 };
	mBounds[1] = { 255, 0 };
	mBounds[2] = { 255, 0 };
	for( auto color : mColorVecs ) {
		// Iterate over all 3 color channels
		for( int i = 0; i < 3; ++i ) {
			if( color[i] < mBounds[i].first )
				mBounds[i].first = color[i];
			if( color[i] > mBounds[i].second )
				mBounds[i].second = color[i];
		}
	}
}

int RgbCube::getSize( ColorChannel channel ) const
{
	size_t idx = static_cast<size_t>( channel );
	int s = mBounds[idx].second - mBounds[idx].first;
	CI_ASSERT( s >= 0 );
	return s;
}

std::pair<uint8_t, uint8_t> RgbCube::getBounds( ColorChannel channel ) const
{
	return mBounds[static_cast<size_t>( channel )];
}

ColorChannel RgbCube::getLongestAxis() const
{
	int sr = getSize( RED );
	int sg = getSize( GREEN );
	int sb = getSize( BLUE );
	
	int m = math<int>::max( math<int>::max( sr, sb ), sb );
	if ( m == sr ) return RED;
	if( m == sg ) return GREEN;
	return BLUE;
}

ci::Color8u RgbCube::calcMeanColor() const
{
	vec3 sum( 0 );
	for( auto& c : mColorVecs ) {
		sum += c;
	}
	sum /= (float) mColorVecs.size();
	
	return Color8u( sum.x, sum.y, sum.z );
}

std::pair<RgbCube, RgbCube> RgbCube::splitAtMedian()
{
	ColorChannel longest = getLongestAxis();
	// Sort the colors according the the specified color channel
	switch( longest ) {
		case RED:
			std::sort( mColorVecs.begin(), mColorVecs.end(), []( const ivec3& c0, const ivec3& c1 ) -> bool { return c0.x < c1.x; } );
			break;
		case GREEN:
			std::sort( mColorVecs.begin(), mColorVecs.end(), []( const ivec3& c0, const ivec3& c1 ) -> bool { return c0.y < c1.y; } );
			break;
		case BLUE:
			std::sort( mColorVecs.begin(), mColorVecs.end(), []( const ivec3& c0, const ivec3& c1 ) -> bool { return c0.z < c1.z; } );
			break;
	}
	
	// Separate at median & subdivide into two new cubes
	CI_ASSERT( mColorVecs.size() >= 2 );
	
	size_t medianIdx = mColorVecs.size() / 2;
	auto cube0 = RgbCube{ std::vector<ivec3>{ mColorVecs.begin(), mColorVecs.begin() + medianIdx } };
	auto cube1 = RgbCube{ std::vector<ivec3>{ mColorVecs.begin() + medianIdx, mColorVecs.end() } };
	return std::pair< RgbCube, RgbCube>{ cube0, cube1 };
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////// COLOR PALETTE GENERATOR /////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

PaletteGenerator::PaletteGenerator( ci::Surface8u surface )
{
	Surface8u::ConstIter iter = surface.getIter();
	while( iter.line() ) {
		while( iter.pixel() ) {
			mColorsVecs.emplace_back( iter.r(), iter.g(), iter.b() );
		}
	}
}

Color8u PaletteGenerator::randomSample() const
{
	size_t idx = (size_t) math<float>::floor( Rand::randFloat() * mColorsVecs.size() );
	auto cv = mColorsVecs.at( idx );
	return Color8u( cv.x, cv.y, cv.z );
}

float getLuminance( const glm::ivec3& c )
{
	return ( c.r + c.r + c.g + c.b + c.b + c.b ) / ( 255.0f * 6.0f );
}

float getLuminance( const Color8u& c )
{
	return float( c.r + c.r + c.g + c.b + c.b + c.b ) / ( 255.0f * 6.0f );
}

std::vector<ci::Color8u> PaletteGenerator::randomPalette( size_t num, float luminanceThreshold ) const
{
	std::vector<ci::Color8u> samples;
	while ( samples.size() < num ) {
		Color8u sample = randomSample();
		if( getLuminance( sample ) >= luminanceThreshold ) {
			samples.emplace_back( sample );
		}
	}
	return samples;
}

std::vector<Color8u> PaletteGenerator::medianCutPalette( size_t num, bool randomize, float luminanceThreshold ) const
{
	std::list<RgbCube> cubes;
	if( luminanceThreshold > 0.0f && luminanceThreshold < 1.0f ) {
		std::vector<ivec3> trimmedColorVecs = mColorsVecs;
		
		auto newEnd = std::remove_if( trimmedColorVecs.begin(), trimmedColorVecs.end(),
									  [luminanceThreshold]( const ivec3& c ) {
										  float luminance = getLuminance( c ) - 0.5f;
										  return( ( 0.5f - math<float>::abs( luminance ) ) < luminanceThreshold );
									  } );
		
		trimmedColorVecs.erase( newEnd, trimmedColorVecs.end() );
		cubes.emplace_back( RgbCube{ trimmedColorVecs } );
	} else {
		cubes.emplace_back( RgbCube{ mColorsVecs } );
	}

	// split the cubes until we get required amount of colors
	while( cubes.size() < num ) {
		auto& parent = cubes.front();
		const auto& pair = parent.splitAtMedian();
		
		// Randomize order non-power-of-two number of colors
		if( randomize && Rand::randBool() ) {
			cubes.emplace_back( pair.second );
			cubes.emplace_back( pair.first );
		} else {
			cubes.emplace_back( pair.first );
			cubes.emplace_back( pair.second );
		}
		cubes.pop_front();
	}
	
	// get the final palette
	std::vector<Color8u> palette;
	for( auto& cube : cubes ) {
		palette.emplace_back( cube.calcMeanColor() );
	}
	return palette;
}

std::vector<ci::Color8u> PaletteGenerator::kmeansPalette( size_t num ) const
{
	
	RgbCube cube( mColorsVecs );
	auto redBounds = cube.getBounds( RED );
	auto greenBounds = cube.getBounds( GREEN);
	auto blueBounds = cube.getBounds( BLUE );
	
	std::vector<ivec3> centroids;
	
	for( size_t i=0; i<num; ++i ) {
		int r = Rand::randInt( redBounds.first, redBounds.second + 1 );
		int g = Rand::randInt( greenBounds.first, greenBounds.second + 1 );
		int b = Rand::randInt( blueBounds.first, blueBounds.second + 1 );
		centroids.emplace_back( r, g, b );
	}

	CI_ASSERT_MSG( false, "Not fully implemented yet..." );

	return {};
}

std::vector<ci::Color8u> randomPalette( ci::Surface8u surface, size_t num )
{
	PaletteGenerator generator( surface );
	return generator.randomPalette( num );
}

