#pragma once
#if defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#endif
namespace radar {
// Diagnostic labels only; failure never changes scheduling or pipeline state.
inline void name_thread(const char *name) noexcept {
#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
#endif
}
} // namespace radar
