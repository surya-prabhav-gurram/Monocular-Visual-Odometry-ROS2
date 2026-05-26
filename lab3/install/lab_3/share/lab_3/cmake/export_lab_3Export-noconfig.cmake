#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lab_3::feature_trackers" for configuration ""
set_property(TARGET lab_3::feature_trackers APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lab_3::feature_trackers PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libfeature_trackers.a"
  )

list(APPEND _cmake_import_check_targets lab_3::feature_trackers )
list(APPEND _cmake_import_check_files_for_lab_3::feature_trackers "${_IMPORT_PREFIX}/lib/libfeature_trackers.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
