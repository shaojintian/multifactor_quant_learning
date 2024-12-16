# cmake/FindUtf8Proc.cmake
find_path(UTF8PROC_INCLUDE_DIR utf8proc.h
  PATHS /usr/local/opt/utf8proc/include
)

find_library(UTF8PROC_LIBRARY
  NAMES utf8proc
  PATHS /usr/local/opt/utf8proc/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Utf8Proc
  REQUIRED_VARS UTF8PROC_LIBRARY UTF8PROC_INCLUDE_DIR
)

if(Utf8Proc_FOUND AND NOT TARGET Utf8Proc::Utf8Proc)
  add_library(Utf8Proc::Utf8Proc UNKNOWN IMPORTED)
  set_target_properties(Utf8Proc::Utf8Proc PROPERTIES
    IMPORTED_LOCATION "${UTF8PROC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${UTF8PROC_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(UTF8PROC_INCLUDE_DIR UTF8PROC_LIBRARY)