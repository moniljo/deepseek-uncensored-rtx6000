#pragma once
#include <cstddef>
#include <limits>
namespace strata::ds4::gpumiss {
constexpr int max_tokens=4, max_picks=6, physical_stride=8, capacity=24;
inline bool parse(const char* value,bool& enabled) {
    if(!value){enabled=false;return true;}
    if(value[0]=='0'&&value[1]==0){enabled=false;return true;}
    if(value[0]=='1'&&value[1]==0){enabled=true;return true;}
    return false;
}
inline bool window(bool enabled,int T){return enabled&&T>=1&&T<=max_tokens;}
enum class Admission{ready,unavailable,invalid};
struct Plan {int unique[capacity]={},pick[physical_stride*max_tokens]={};int count=0,entries=0,resident=0;};
inline Admission plan(int T,int k,int n,const int* ids,const int* slot,const int* ram,Plan& out) {
    if(T<1||T>max_tokens||k<1||k>max_picks||n<1||n>1024||!ids||!slot||!ram)return Admission::unavailable;
    Plan next;for(int& p:next.pick)p=-1;
    for(int t=0;t<T;++t)for(int i=0;i<k;++i) {
        const int e=ids[t*physical_stride+i];if(e<0||e>=n)return Admission::invalid;
        if(slot[e]>=0){++next.resident;continue;}
        if(ram[e]<0)return Admission::unavailable;
        int at=0;while(at<next.count&&next.unique[at]!=e)++at;
        if(at==next.count){if(next.count==capacity)return Admission::unavailable;next.unique[next.count++]=e;}
        next.pick[t*physical_stride+i]=at;++next.entries;
    }
    out=next;return Admission::ready;
}
inline bool sizes(std::size_t blob,std::size_t& stride,std::size_t& bytes) {
    if(!blob||blob>std::numeric_limits<std::size_t>::max()-4095)return false;
    const std::size_t s=(blob+4095)&~std::size_t(4095);
    if(s>std::numeric_limits<std::size_t>::max()/capacity)return false;
    stride=s;bytes=s*capacity;return true;
}
} // namespace strata::ds4::gpumiss
