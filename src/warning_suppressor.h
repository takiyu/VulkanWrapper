// -----------------------------------------------------------------------------
// ------------- Warning suppression for third party include (push) ------------
// -----------------------------------------------------------------------------
#if defined __clang__
#define VKW_SUPPRESS_WARNING_PUSH \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Weverything\"")
#elif defined __GNUC__
#define VKW_SUPPRESS_WARNING_PUSH \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
    _Pragma("GCC diagnostic ignored \"-Wctor-dtor-privacy\"") \
    _Pragma("GCC diagnostic ignored \"-Wshadow\"") \
    _Pragma("GCC diagnostic ignored \"-Wmissing-declarations\"") \
    _Pragma("GCC diagnostic ignored \"-Wstrict-overflow\"") \
    _Pragma("GCC diagnostic ignored \"-Wdeprecated-copy\"") \
    _Pragma("GCC diagnostic ignored \"-Wnoexcept\"") \
    _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"")
#elif defined _MSC_VER
    // TODO: MSVC here
#else
#define VKW_SUPPRESS_WARNING_PUSH  // empty
#endif

// -----------------------------------------------------------------------------
// ------------- Warning suppression for third party include (pop) -------------
// -----------------------------------------------------------------------------
#ifdef __clang__
#define VKW_SUPPRESS_WARNING_POP \
    _Pragma("clang diagnostic pop")
#elif __GNUC__
#define VKW_SUPPRESS_WARNING_POP \
    _Pragma("GCC diagnostic pop")
#elif _MSC_VER
    // TODO: MSVC here
#else
#define VKW_SUPPRESS_WARNING_POP  // empty
#endif
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
