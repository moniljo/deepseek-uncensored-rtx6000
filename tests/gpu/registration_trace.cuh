// Fixture-only API observation/failure injection. Successful calls still use
// REAL CUDA registration; no production policy or expert math is mirrored.
#pragma once
#include <array>
#include <atomic>

struct RegistrationTrace {
    std::atomic<int>registrations{0},unregistrations{0},meminfo_calls{0};
    int fail_registration=0,fail_unregistration=0,fail_meminfo=0;
    bool stop_after_registration=false;
    cudaError_t registration_error=cudaErrorMemoryAllocation;
    std::array<uintptr_t,64> registered_addresses{},unregistered_addresses{};
    std::array<size_t,64> registered_lengths{};
    std::atomic<int>successes{0},releases{0};
    void reset(){
        registrations=0;unregistrations=0;meminfo_calls=0;successes=0;releases=0;
        fail_registration=fail_unregistration=fail_meminfo=0;stop_after_registration=false;
        registration_error=cudaErrorMemoryAllocation;
        registered_addresses.fill(0);unregistered_addresses.fill(0);registered_lengths.fill(0);
    }
}registration_api;

cudaError_t fixture_host_register(void* base,size_t bytes,unsigned flags){
    const int call=++registration_api.registrations;
    if(call==registration_api.fail_registration)return registration_api.registration_error;
    if(registration_api.successes>=(int)registration_api.registered_addresses.size())return cudaErrorInvalidValue;
    const auto result=cudaHostRegister(base,bytes,flags);
    if(result==cudaSuccess){
        const size_t at=(size_t)registration_api.successes++;
        registration_api.registered_addresses[at]=(uintptr_t)base;
        registration_api.registered_lengths[at]=bytes;
    }
    return result;
}
cudaError_t fixture_host_unregister(void* base){
    const int call=++registration_api.unregistrations;
    if(call==registration_api.fail_unregistration)return cudaErrorInvalidValue;
    if(registration_api.releases>=(int)registration_api.unregistered_addresses.size())return cudaErrorInvalidValue;
    const auto result=cudaHostUnregister(base);
    if(result==cudaSuccess)registration_api.unregistered_addresses[(size_t)registration_api.releases++]=(uintptr_t)base;
    return result;
}
cudaError_t fixture_registration_meminfo(size_t* available,size_t* total){
    if(++registration_api.meminfo_calls==registration_api.fail_meminfo)return cudaErrorInvalidValue;
    return cudaMemGetInfo(available,total);
}
