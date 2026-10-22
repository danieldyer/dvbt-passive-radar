#pragma once
// Bulk binary IPC. libc++'s std::cin streambuf on macOS can service large
// xsgetn requests through a locked getc for every byte, even with sync off.
#include <cerrno>
#include <cstddef>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace radar::support {
inline bool read_stdin(void* destination,std::size_t count,bool allow_eof=false){
    auto* bytes=static_cast<char*>(destination);std::size_t done=0;
    while(done<count){
        const auto n=::read(STDIN_FILENO,bytes+done,count-done);
        if(n<0){if(errno==EINTR)continue;throw std::system_error(errno,std::generic_category(),"Read binary input");}
        if(n==0){if(allow_eof && done==0)return false;throw std::runtime_error("Truncated binary input");}
        done+=std::size_t(n);
    }
    return true;
}
inline void write_stdout(const void* source,std::size_t count){
    const auto* bytes=static_cast<const char*>(source);std::size_t done=0;
    while(done<count){
        const auto n=::write(STDOUT_FILENO,bytes+done,count-done);
        if(n<0){if(errno==EINTR)continue;throw std::system_error(errno,std::generic_category(),"Write binary output");}
        if(n==0)throw std::runtime_error("Closed binary output");
        done+=std::size_t(n);
    }
}
}
