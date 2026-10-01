#include "Semihosting.hpp"
#include <cerrno>
#include <cstdio>
#include <cwchar>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

extern char _end;
extern char _heap_end;

extern "C"
{
    void* __dso_handle = nullptr;

    caddr_t _sbrk(int incr)
    {
        static char* heap = &_end;
        char* prev = heap;
        incr = (incr + 3) & ~3;
        if (heap + incr > &_heap_end)
        {
            errno = ENOMEM;
            return reinterpret_cast<caddr_t>(-1);
        }
        heap += incr;
        return reinterpret_cast<caddr_t>(prev);
    }

    void _exit(int status)
    {
        platform::qemu::SemihostingExit(status);
    }

    [[gnu::weak]] void Default_Handler_Forwarded()
    {
        std::fflush(stdout);
        _exit(1);
    }

    void abort()
    {
        std::fflush(stdout);
        _exit(1);
    }

    int _fstat(int, struct stat* st)
    {
        st->st_mode = S_IFCHR;
        return 0;
    }

    void _init()
    {}

    void _fini()
    {}

    void HardwareInitialization()
    {}

    char* getcwd(char* buf, size_t size)
    {
        if (buf == nullptr || size < 2)
        {
            errno = ERANGE;
            return nullptr;
        }

        buf[0] = '/';
        buf[1] = '\0';
        return buf;
    }

    int mkdir(const char*, mode_t)
    {
        errno = ENOSYS;
        return -1;
    }

    int swprintf(wchar_t*, size_t, const wchar_t*, ...)
    {
        return -1;
    }
}
