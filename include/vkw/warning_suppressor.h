#ifndef WARNING_SUPPRESSOR_191222
#define WARNING_SUPPRESSOR_191222

// -----------------------------------------------------------------------------
// ------------ Warning suppression for third party include (begin) ------------
// -----------------------------------------------------------------------------
#if defined __clang__
#define BEGIN_VKW_SUPPRESS_WARNING \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Weverything\"")
#elif defined __GNUC__
#define BEGIN_VKW_SUPPRESS_WARNING \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
    _Pragma("GCC diagnostic ignored \"-Wctor-dtor-privacy\"") \
    _Pragma("GCC diagnostic ignored \"-Wshadow\"") \
    _Pragma("GCC diagnostic ignored \"-Wmissing-declarations\"") \
    _Pragma("GCC diagnostic ignored \"-Wstrict-overflow\"") \
    _Pragma("GCC diagnostic ignored \"-Wdeprecated-copy\"") \
    _Pragma("GCC diagnostic ignored \"-Wnoexcept\"") \
    _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"") \
    _Pragma("GCC diagnostic ignored \"-Winit-list-lifetime\"")
#elif defined _MSC_VER
#define BEGIN_VKW_SUPPRESS_WARNING  // TODO: MSVC here
#else
#define BEGIN_VKW_SUPPRESS_WARNING  // empty
#endif

// -----------------------------------------------------------------------------
// ------------- Warning suppression for third party include (end) -------------
// -----------------------------------------------------------------------------
#ifdef __clang__
#define END_VKW_SUPPRESS_WARNING \
    _Pragma("clang diagnostic pop")
#elif __GNUC__
#define END_VKW_SUPPRESS_WARNING \
    _Pragma("GCC diagnostic pop")
#elif _MSC_VER
#define END_VKW_SUPPRESS_WARNING  // TODO: MSVC here
#else
#define END_VKW_SUPPRESS_WARNING  // empty
#endif
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#endif // end of include guard
