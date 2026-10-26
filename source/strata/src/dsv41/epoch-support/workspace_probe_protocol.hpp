#pragma once
#include "pool_probe_protocol.hpp"
namespace strata::ds4::workspaceprobe {
inline bool is_command(const std::string& line) {
    return line=="WORKSPACE_REGROW"||line.rfind("WORKSPACE_REGROW ",0)==0;
}
inline bool command(const std::string& line,int& timeout) {
    timeout=0;constexpr size_t prefix=sizeof("WORKSPACE_REGROW ")-1;
    if(line.rfind("WORKSPACE_REGROW ",0)!=0||line.size()<prefix+1||line.size()>prefix+5)return false;
    return probeproto::command("EPOCH "+line.substr(prefix),timeout);
}
inline const char* result_marker(bool terminal,bool operation_ok,bool published) {
    return terminal?"WORKSPACE_REGROW_FAILED":published?"WORKSPACE_REGROW_PUBLISHED":
           operation_ok?"WORKSPACE_REGROW_KEPT_SMALL":"WORKSPACE_REGROW_REFUSED";
}
inline const char* resume_marker(bool workspace,bool pool,bool ok) {
    return workspace?(ok?"WORKSPACE_REGROW_RESUMED":"WORKSPACE_REGROW_RESUME_FAILED"):
                     poolprobe::resume_marker(pool,ok);
}
} // namespace strata::ds4::workspaceprobe
