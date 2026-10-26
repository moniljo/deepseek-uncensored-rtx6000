#pragma once
// Native diagnostic-only syntax/markers. Existing EPOCH parser stays unchanged.
#include "probe_protocol.hpp"
namespace strata::ds4::poolprobe {
inline bool is_command(const std::string& line) {
    return line=="POOL_SHRINK"||line.rfind("POOL_SHRINK ",0)==0;
}
inline bool command(const std::string& line,int& timeout) {
    timeout=0;
    constexpr size_t prefix=sizeof("POOL_SHRINK ")-1;
    if(line.rfind("POOL_SHRINK ",0)!=0||line.size()<prefix+1||line.size()>prefix+5)return false;
    return probeproto::command("EPOCH "+line.substr(prefix),timeout);
}
inline const char* result_marker(bool ok,bool full) {
    return !ok?"POOL_SHRINK_FAILED":full?"POOL_SHRINK_DONE":"POOL_SHRINK_PARTIAL";
}
inline const char* resume_marker(bool pool,bool ok) {
    return pool?(ok?"POOL_SHRINK_RESUMED":"POOL_SHRINK_RESUME_FAILED"):
                (ok?"EPOCH_RESUMED":"EPOCH_RESUME_FAILED");
}
} // namespace strata::ds4::poolprobe
