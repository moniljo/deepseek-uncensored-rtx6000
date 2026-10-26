// Experimental diagnostic command parsing/formatting only. No model/CUDA calls.
#pragma once
#include <cstring>
#include <sstream>
#include <string>
namespace strata::ds4::probeproto {
inline bool flag(const char* value,bool& enabled) {
    enabled=false;
    if(!value)return true;
    if(std::strcmp(value,"0")!=0&&std::strcmp(value,"1")!=0)return false;
    enabled=value[0]=='1';return true;
}
inline bool is_command(const std::string& line) {return line=="EPOCH"||line.rfind("EPOCH ",0)==0;}
inline bool command(const std::string& line,int& timeout) {
    timeout=0;if(line.rfind("EPOCH ",0)!=0||line.size()<7||line.size()>11)return false;
    unsigned n=0;
    for(size_t i=6;i<line.size();++i){const char c=line[i];if(c<'0'||c>'9')return false;n=n*10+unsigned(c-'0');if(n>60000)return false;}
    if(!n)return false;timeout=int(n);return true;
}
inline std::string quote(const std::string& s) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:s){if(c=='"'||c=='\\')out<<'\\'<<char(c);else if(c=='\n')out<<"\\n";
        else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";else out<<char(c<32?'?':c);}
    out<<'"';return out.str();
}
} // namespace strata::ds4::probeproto
