#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomma"


/**
  
 * \file lifFile.cpp
 * \brief Define classes to read directly from Leica .lif file
 * \author Mathieu Leocmach
 * \date 28 March 2009
 *
 * Strongly inspired from BioimageXD VTK implementation but using standard library instead of VTK objects
 *
 */
#include "otherIO/lifFile.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>

namespace  {
    
    struct ifStreamDeleter
    {
        void operator()(std::ifstream* p) const
        {
            if ( p != nullptr )
            {
                p->close();
                delete p;
            }
        }
    };
    std::shared_ptr<std::ifstream> make_shared_ifstream(std::ifstream * ifstream_ptr)
    {
        return std::shared_ptr<std::ifstream>(ifstream_ptr, ifStreamDeleter());
    }
    std::shared_ptr<std::ifstream> make_shared_ifstream(std::string filename)
    {
        return make_shared_ifstream(new std::ifstream(filename, std::ifstream::in | std::ifstream::binary));
    }
}

using namespace std;


/** @brief LifSerieHeader constructor  */
lifIO::LifSerieHeader::LifSerieHeader(TiXmlElement *root) : name(root->Attribute("Name")), rootElement(root)
{
    //XML parsing
    TiXmlNode *elementImage = rootElement->FirstChild("Data")->FirstChild("Image");
    // If Image element found
    if (elementImage)
        parseImage(elementImage);
    
    m_cached_frame_duration.first = false;
    return;
}


/** \brief get the real size of a pixel (in meters) in the dimension d */
double lifIO::LifSerieHeader::getVoxelSize(const size_t d) const
{
    const string voxel = "dblVoxel", dims = "XYZ";
    map<string, ScannerSettingRecord>::const_iterator it = scannerSettings.find(voxel+dims[d]);
    if(it == scannerSettings.end())
        return 0.0;
    stringstream sstream( it->second.variant);
    double voxelSize;
    sstream >> voxelSize;
    return voxelSize;
}

/** \brief get the resolution of a channel of a serie, in Bits per pixel*/
int lifIO::LifSerieHeader::getResolution(const size_t channel) const
{
    return channels[channel].resolution;
}

/** @brief get the number of time steps  */
size_t lifIO::LifSerieHeader::getNbTimeSteps() const
{
    map<string, DimensionData>::const_iterator it = dimensions.find("T");
    if(it == dimensions.end()) return 1;
    return it->second.numberOfElements;
}

bool lifIO::LifSerieHeader::hasTimeChannel() const
{
    map<string, DimensionData>::const_iterator it = dimensions.find("T");
    return it != dimensions.end();
}

/** @brief getSpatialDimensions  */
vector<size_t> lifIO::LifSerieHeader::getSpatialDimensions() const
{
    vector<size_t> dims;
    for(map<string, DimensionData>::const_iterator d = dimensions.begin(); d != dimensions.end(); d++)
        if(d->second.dimID <4)
			dims.push_back(d->second.numberOfElements);
    return dims;
}

/** @brief get the number of pixels in one time step, ie the product of the spatial dimensions  */
size_t lifIO::LifSerieHeader::getNbPixelsInOneTimeStep() const
{
    const vector<size_t> dims = getSpatialDimensions();
    return accumulate(dims.begin(), dims.end(), 1, multiplies<size_t>());
}

/** @brief get the number of pixels in one slice, ie the product of the two first spatial dimensions  */
size_t lifIO::LifSerieHeader::getNbPixelsInOneSlice() const
{
    vector<size_t> dims = getSpatialDimensions();
    dims.resize(2, 1);
    return accumulate(dims.begin(), dims.end(), 1, multiplies<size_t>());
}

/** @brief get the ratio (Z voxel size)/(X voxel size)  */
double lifIO::LifSerieHeader::getZXratio() const
{
    double x = getVoxelSize(0), z = getVoxelSize(2);
    if(z==0.0)
        return 1.0;
    else
        return z/x;
}



/** @brief parse the "Image" node of the XML header  */
void lifIO::LifSerieHeader::parseImage(TiXmlNode *elementImage)
{
    TiXmlNode *elementImageDescription = elementImage->FirstChild("ImageDescription");
    if (elementImageDescription)
        parseImageDescription(elementImageDescription);
    // Parse time stamps even if there aren't any, then add empty
    // Unsigned Long Long to timestamps vector
    TiXmlNode *elementTimeStampList = elementImage->FirstChild("TimeStampList");
    parseTimeStampList(elementTimeStampList);
    // Parse Hardware Setting List even if there aren't
    TiXmlNode *elementHardwareSettingList = 0;
    TiXmlNode *child = 0;
    string Name;
    while( (child = elementImage->IterateChildren("Attachment", child)) )
    {
        Name = child->ToElement()->Attribute("Name");
        if(Name=="HardwareSettingList" && !child->NoChildren())
            elementHardwareSettingList = child;
    }
    if(elementHardwareSettingList)
    	parseHardwareSettingList(elementHardwareSettingList);
}

/** @brief parse the "ImageDescription" node of the XML header  */
void lifIO::LifSerieHeader::parseImageDescription(TiXmlNode *elementImageDescription)
{
    TiXmlNode *elementChannels = elementImageDescription->FirstChild("Channels");
    TiXmlNode *elementDimensions = elementImageDescription->FirstChild("Dimensions");
    if (elementChannels && elementDimensions)
	{
        TiXmlNode *child = 0;
        // Get info of channels
        while((child = elementChannels->IterateChildren(child)))
            this->channels.push_back(ChannelData(child->ToElement()));
        // Get info of dimensions
        child = 0;
        while((child = elementDimensions->IterateChildren(child)))
        {
            DimensionData data(child->ToElement());
            this->dimensions.insert(
                make_pair(data.getName(),data)
                );
        }
	}
}

/** @brief parse the "TimeStampList" node of the XML header  */
void lifIO::LifSerieHeader::parseTimeStampList(TiXmlNode *elementTimeStampList)
{
    if (elementTimeStampList)
    {
        int NumberOfTimeStamps = 0;
        //new way to store timestamps
        if (elementTimeStampList->ToElement()->Attribute("NumberOfTimeStamps", &NumberOfTimeStamps) != 0 && NumberOfTimeStamps != 0)
        {
            this->timeStamps.resize(NumberOfTimeStamps);
            //timestamps are stored in the text of the node as 16bits hexadecimal separated by spaces
            //transform this node text in a stream
            std::istringstream in(elementTimeStampList->ToElement()->GetText());
            //convert each number from hex to unsigned long long and fill in the timestamp vector
            copy(
                std::istream_iterator<unsigned long long>(in >> std::hex),
                std::istream_iterator<unsigned long long>(),
                this->timeStamps.begin());
        }
        
        //old way to store time stamps
        else
        {
            unsigned long highInt;
            unsigned long lowInt;
            unsigned long long timeStamp;
            TiXmlNode *child = 0;
            while((child = elementTimeStampList->IterateChildren(child)))
            {
                child->ToElement()->QueryValueAttribute<unsigned long>("HighInteger",&highInt);
                child->ToElement()->QueryValueAttribute<unsigned long>("LowInteger",&lowInt);
                timeStamp = highInt;
                timeStamp <<= 32;
                timeStamp += lowInt;
                timeStamp /= 10000; // Convert to ms
                this->timeStamps.push_back(timeStamp);
            }
        }
    }
}

/** @brief parse the "HardwareSettingList" node of the XML header  */
void lifIO::LifSerieHeader::parseHardwareSettingList(TiXmlNode *elementHardwareSettingList)
{
    TiXmlNode *elementHardwareSetting = elementHardwareSettingList->FirstChild();
    TiXmlNode *elementScannerSetting = elementHardwareSetting->FirstChild("ScannerSetting");
    parseScannerSetting(elementScannerSetting);
    //no real need of filter settings for the moment
    //TiXmlNode *elementFilterSetting = elementHardwareSetting->FirstChild("FilterSetting");
    //ParseFilterSetting(elementFilterSetting);
}

/** @brief parse the "ScannerSetting" node of the XML header  */
void lifIO::LifSerieHeader::parseScannerSetting(TiXmlNode *elementScannerSetting)
{
    if(elementScannerSetting)
    {
        TiXmlNode *child = 0;
        while((child = elementScannerSetting->IterateChildren("ScannerSettingRecord", child)))
            this->scannerSettings.insert(
                make_pair(
                    child->ToElement()->Attribute("Identifier"),
                    ScannerSettingRecord(child->ToElement())
                    )
                );
    }
}

/**
 \brief caclulate frame_duration average
 */
float lifIO::LifSerieHeader::frame_duration_ms() const
{
    if (m_cached_frame_duration.first == false)
    {
        float avg_fd = 0.0;
        if (getTimestamps().size())
            {
                m_frame_durations.resize(getTimestamps().size());
                adjacent_difference(getTimestamps().begin(), getTimestamps().end(), m_frame_durations.begin());
                m_frame_durations[0] = 0; // first difference not written
                auto sumall = accumulate(m_frame_durations.begin(), m_frame_durations.end(), 0);
                avg_fd = (sumall / (m_frame_durations.size()-1))/10000.0f;
            }
        m_cached_frame_duration.first = true;
        m_cached_frame_duration.second = avg_fd;
    }
    return m_cached_frame_duration.second;
    
}

/** @brief LifSerie constructor  */
lifIO::LifSerie::LifSerie(LifSerieHeader serie, const std::string &filename,
                          unsigned long long offset, unsigned long long memorySize) : LifSerieHeader(serie)
{
    fileRef = make_shared_ifstream(filename.c_str());
    if(! fileRef || !fileRef->is_open())
        throw invalid_argument(("No such file as "+filename).c_str());

    //get size of the file
    fileRef->seekg(0,ios::end);
    fileSize = fileRef->tellg();

    //check the validity of the offset and memorysize parameters
    if(offset >= (unsigned long long)fileSize)
        throw invalid_argument("Offset is larger than file size");
    if(offset+memorySize > (unsigned long long)fileSize)
        throw invalid_argument("The end of the serie is further than the end of file");

    this->offset = offset;
    this->memorySize = memorySize;
}



/**
    \brief fill a memory buffer that already has the good dimension
    If more than one channel, all channels are retrieved interlaced
  */
void lifIO::LifSerie::fill3DBuffer(void* buffer, size_t t) const
{
    char *pos = static_cast<char*>(buffer);
    unsigned long int frameDataSize = getNbPixelsInOneTimeStep()*channels.size();
    fileRef->seekg(getOffset(t) ,ios::beg);
    fileRef->read(pos,frameDataSize);
}

/**
    \brief fill a memory buffer that already has the good dimension
    If more than one channel, all channels are retrieved interlaced
  */
void lifIO::LifSerie::fill2DBuffer(void* buffer, size_t t, size_t z) const
{
    char *pos = static_cast<char*>(buffer);
    unsigned long int sliceDataSize = getNbPixelsInOneSlice()*channels.size();
    fileRef->seekg(getOffset(t) + z *  sliceDataSize, ios::beg);
    fileRef->read(pos, sliceDataSize);
}

/** @brief return an iterator to the begining of the data of time step t
    No gestion of multi-channel.
*/
istreambuf_iterator<char> lifIO::LifSerie::begin(size_t t)
{
    fileRef->seekg(getOffset(t));
    return istreambuf_iterator<char>(*fileRef);
}


/** @brief get the position in file where starts the data of time step t  */
unsigned long long lifIO::LifSerie::getOffset(size_t t) const
{
    if(t >= getNbTimeSteps())
        throw out_of_range("Time step out of range");
    if(t==0)
        return this->offset;

    map<string, DimensionData>::const_iterator it = dimensions.find("T");
    if(it == dimensions.end())
        return this->offset;

    return this->offset + it->second.bytesInc * t;
}

/** @brief LifHeader constructor  */
 lifIO::LifHeader::LifHeader(TiXmlDocument &header)
{
    this->m_xmldoc = header;
    parseHeader();
    return;
}

/** @brief LifHeader
  *
  * @todo: document this function
  */
 lifIO::LifHeader::LifHeader(std::string &header)
{
    this->m_xmldoc.Parse(header.c_str(),0);
    parseHeader();
    return;
}



/** @brief parse the XML header  */
void lifIO::LifHeader::parseHeader()
{
    m_xmldoc.RootElement()->QueryIntAttribute("Version", &lifVersion);

    TiXmlElement *elementElement = m_xmldoc.RootElement()->FirstChildElement();
    if (!elementElement)
        throw logic_error("No element Element found after root element.");

    this->name = elementElement->Attribute("Name");
    TiXmlNode *elementChildren = elementElement->FirstChild("Children");
    if (!elementChildren || elementChildren->NoChildren())
        throw logic_error("No serie in this experiment.");

    TiXmlNode *serieNode = 0;
    while( (serieNode = elementChildren->IterateChildren("Element", serieNode)))
    {
        //have to remove some nodes also named "Element" introduced in later versions of LIF
        std::string elname(serieNode->ToElement()->Attribute("Name"));
        if (elname == "BleachPointROISet") continue;
        m_series.push_back(std::make_unique<LifSerieHeader>(serieNode->ToElement()));
    }
}



/** \brief Constructor from lif file name
 */
lifIO::LifReader::LifReader(const string &filename)
{
    const int MemBlockCode = 0x70, TestCode = 0x2a;
    char lifChar;
    m_path = filename;
    
    bool ok = false;
    
    m_fileRef = make_shared_ifstream(filename);
    ok = (m_fileRef &&  m_fileRef->is_open());
    if (! m_fileRef){
        m_Valid = false;
        return;
    }
    
    // Get size of the file
    m_fileRef->seekg(0,ios::end);
    m_lif_file_size = m_fileRef->tellg();
    m_fileRef->seekg(0,ios::beg);
    char buffer[4];

    m_fileRef->read(buffer,4);
    int32_t ibuf = *((int*)(buffer));
    ok &= (MemBlockCode == ibuf);
    
    
    // Skip the size of next block
    m_fileRef->seekg(4,ios::cur);
    *m_fileRef>>lifChar;
    ok &= (lifChar == TestCode);
    
    if (!ok){
        m_Valid = false;
        return;
    }
    m_Valid = ok;
    
    if (! m_Valid) return;
    
    unsigned int xmlChars = readUnsignedInt();
    xmlChars*=2;

    // Read and parse xml header
    string xmlString;
    xmlString.reserve(xmlChars/2);
    {
        std::shared_ptr<char> xmlHeader (new char[xmlChars]);
        m_fileRef->read(xmlHeader.get(),xmlChars);
        for(unsigned int p=0;p<xmlChars/2;++p)
            xmlString.push_back(xmlHeader.get()[2*p]);
    }
    m_header = std::shared_ptr<LifHeader>(new LifHeader(xmlString));
    size_t s = 0;
    while (m_fileRef->tellg() < m_lif_file_size)
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        
        // Check LIF test value
        int lifCheck = readInt();
        if (lifCheck != MemBlockCode){
            m_Valid = false;
            break;
        }

        // Don't care about the size of the next block
        m_fileRef->seekg(4,ios::cur);
        // Read testcode
        *m_fileRef>>lifChar;
        if (lifChar != TestCode){
            m_Valid = false;
            break;
        }

        // Read size of memory, this is 4 bytes in version 1 and 8 in version 2
        unsigned long long memorySize;
        if (this->getVersion() >= 2)
            memorySize = readUnsignedLongLong();
        else
            memorySize = readUnsignedInt();


        // Find next testcode
        lifChar=0;
        while (lifChar != TestCode) {*m_fileRef>>lifChar;}
        unsigned int memDescrSize = readUnsignedInt() * 2;
        // Skip over memory description
        m_fileRef->seekg(memDescrSize,ios::cur);

        // Add serie if memory size is > 0
        if (memorySize > 0)
        {
            if(s >= getLifHeader().getNbSeries()){
                m_Valid = false;
                break;
            }

            m_series.push_back(std::make_unique<LifSerie>(
                    this->m_header->getSerieHeader(s),
                    filename,
                                          m_fileRef->tellg(),
                    memorySize));
            s++;
            //jump to the next memory block
            m_fileRef->seekg(static_cast<streampos>(memorySize),ios::cur);
        }
    }
}

/** \brief read an int form file advancing the cursor*/
int lifIO::LifReader::readInt()
{
    char buffer[4];
    m_fileRef->read(buffer,4);
    return *((int*)(buffer));
}
/** \brief read an unsigned int form file advancing the cursor*/
unsigned int lifIO::LifReader::readUnsignedInt()
{
    char buffer[4];
    m_fileRef->read(buffer,4);
    return *((unsigned int*)(buffer));
}
/** \brief read an unsigned long long int form file advancing the cursor*/
unsigned long long lifIO::LifReader::readUnsignedLongLong()
{
    char buffer[8];
    m_fileRef->read(buffer,8);
    return *((unsigned long long*)(buffer));
}

/** \brief read an unsigned long long int form file advancing the cursor*/
void lifIO::LifReader::close_file()
{
    if(m_fileRef && m_fileRef->is_open())
        m_fileRef->close();
}

/** \brief constructor from XML  */
lifIO::ChannelData::ChannelData(TiXmlElement *element)
{
    element->QueryIntAttribute("DataType",&dataType);
    element->QueryIntAttribute("ChannelTag",&channelTag);
    element->QueryIntAttribute("Resolution",&resolution);
    nameOfMeasuredQuantity = element->Attribute("NameOfMeasuredQuantity");
    element->QueryDoubleAttribute("Min",&minimum);
    element->QueryDoubleAttribute("Max",&maximum);
    unit = element->Attribute("Unit");
    LUTName = element->Attribute("LUTName");
    element->QueryIntAttribute("IsLUTInverted",(int*)&isLUTInverted);
    // Bytes Inc is 64 bits in LIF version 2 but GetScalarAttribute only allows
    // maximum of unsigned long which can be 32 bits.
    element->QueryValueAttribute<unsigned long long>("BytesInc",&bytesInc);
    element->QueryIntAttribute("BitInc",&bitInc);
    
}

/** \brief constructor from XML  */
lifIO::DimensionData::DimensionData(TiXmlElement *element)
{
    element->QueryIntAttribute("DimID",&dimID);
    element->QueryIntAttribute("NumberOfElements", &numberOfElements);
    element->QueryDoubleAttribute("Origin",&origin);
    element->QueryDoubleAttribute("Length",&length);
    unit = element->Attribute("Unit");
    // Bytes Inc is 64 bits in LIF version 2 but GetScalarAttribute only allows
    // maximum of unsigned long which can be 32 bits.
    element->QueryValueAttribute<unsigned long long>("BytesInc",&bytesInc);
    element->QueryIntAttribute("BitInc",&bitInc);
}

/** \brief constructor from XML  */
lifIO::ScannerSettingRecord::ScannerSettingRecord(TiXmlElement *element)
{
    identifier = element->Attribute("Identifier");
    unit = element->Attribute("Unit");
    description = element->Attribute("Description");
    element->QueryIntAttribute("Data",&data);
    variant = element->Attribute("Variant");
    element->QueryIntAttribute("VariantType",&variantType);
}

/** \brief constructor from XML  */
lifIO::FilterSettingRecord::FilterSettingRecord(TiXmlElement *element)
{
    objectName = element->Attribute("ObjectName");
    className = element->Attribute("ClassName");
    attribute = element->Attribute("Attribute");
    description = element->Attribute("Description");
    element->QueryIntAttribute("Data",&data);
    variant = element->Attribute("Variant");
    element->QueryIntAttribute("VariantType",&variantType);
}

/** @brief operator<<
  *
  * @todo: document this function
  */
ostream &lifIO::operator<<(ostream& out, const LifSerieHeader &s)
{
    out<< s.getName() <<":\t";
    const map<string, DimensionData> & dims = s.getDimensionsData();
    out<< dims.size()<<" dimensions\t";
    for(map<string, DimensionData>::const_iterator it=dims.begin(); it!=dims.end();++it)
        out<<it->second<<", ";
    out<<endl;
    out<<"\tVoxel sizes (nm)\t";
    for(map<string, DimensionData>::const_iterator it=dims.begin();it!=dims.end();++it)
        if(it->second.dimID < 4)
            cout<<s.getVoxelSize(it->second.dimID-1)*1E9<<"\t";
    out<<endl;
    const vector<ChannelData> &channels = s.getChannels();
    for( const auto& cc : channels) out << cc.getName() << "\t" << std::endl;
    return out;
}

/** @brief operator<<
  *
  * @todo: document this function
  */
ostream &lifIO::operator<<(ostream& out, const LifHeader &r)
{
    out<< r.getName() <<endl;
    for(size_t s=0; s<r.getNbSeries(); ++s)
        out<<"("<<s<<") "<<r.getSerieHeader(s)<<endl;
    return out;
}

/** @brief operator<<
  *
  * @todo: document this function
  */
ostream &lifIO::operator<<(ostream& out, const DimensionData &d)
{
    out<<d.getName()<<" "<<d.numberOfElements;
    return out;
}

#pragma GCC diagnostic pop

