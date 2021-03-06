
#include "logger/logger.hpp"
#include "core/stl_utils.hpp"
#include <string>
#include  <iomanip>

namespace logging {
std::string file_path_by_appending_component(const std::string& path, const std::string& component, bool is_directory)
{
    // FIXME: Does this have to be changed to accomodate Windows platforms?
    std::string buffer;
    buffer.reserve(2 + path.length() + component.length());
    buffer.append(path);
    std::string terminal = "";
    if (is_directory && component[component.length() - 1] != '/') {
        terminal = "/";
    }
    char path_last = path[path.length() - 1];
    char component_first = component[0];
    if (path_last == '/' && component_first == '/') {
        buffer.append(component.substr(1));
        buffer.append(terminal);
    } else if (path_last == '/' || component_first == '/') {
        buffer.append(component);
        buffer.append(terminal);
    } else {
        buffer.append("/");
        buffer.append(component);
        buffer.append(terminal);
    }
    return buffer;
}

std::string file_path_by_appending_extension(const std::string& path, const std::string& extension)
{
    std::string buffer;
    buffer.reserve(1 + path.length() + extension.length());
    buffer.append(path);
    char path_last = path[path.length() - 1];
    char extension_first = extension[0];
    if (path_last == '.' && extension_first == '.') {
        buffer.append(extension.substr(1));
    } else if (path_last == '.' || extension_first == '.') {
        buffer.append(extension);
    } else {
        buffer.append(".");
        buffer.append(extension);
    }
    return buffer;
}

std::string create_timestamped_template(const std::string& prefix, int wildcard_count)
{
    constexpr std::pair<int,int> WILDCARD_RANGE (6,10);
    wildcard_count = std::min(WILDCARD_RANGE.first, std::max(WILDCARD_RANGE.second, wildcard_count));
    std::time_t time = std::time(nullptr);
    std::stringstream stream;
    struct tm time_spot;
    localtime_r(&time, &time_spot);
    stream << prefix << "-" << std::put_time(&time_spot, "%Y%m%d-%H%M%S") << "-" << std::string(wildcard_count, 'X');
    return stream.str();
}

std::string reserve_unique_file_name(const std::string& path, const std::string& template_string)
{
    assert (template_string.find("XXXXXX") != std::string::npos);
    std::string path_buffer = file_path_by_appending_component(path, template_string, false);
    int fd = mkstemp(&path_buffer[0]);
    if (fd < 0) {
        int err = errno;
        throw std::system_error(err, std::system_category());
    }
    // Remove the file so we can use the name for our own file.
    close(fd);
    unlink(path_buffer.c_str());
    return path_buffer;
}


}

