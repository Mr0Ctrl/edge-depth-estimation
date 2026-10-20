cmake_minimum_required(VERSION 3.16)
project(SpatialEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}")

# Core library sources (in src/)
file(GLOB_RECURSE SPATIAL_CORE_SOURCES "src/*.cpp")

if(SPATIAL_CORE_SOURCES)
    add_library(spatial_core ${SPATIAL_CORE_SOURCES})
    target_include_directories(spatial_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(spatial_core PUBLIC ${OpenCV_LIBS})
    message(STATUS "Configuring spatial_core library with: ${SPATIAL_CORE_SOURCES}")
else()
    message(STATUS "No files in src/ yet. Write your .cpp files in src/ to build spatial_core.")
endif()

# Executable apps (in apps/)
file(GLOB_RECURSE SPATIAL_APP_SOURCES "apps/*.cpp")

foreach(APP_SRC ${SPATIAL_APP_SOURCES})
    get_filename_component(APP_NAME ${APP_SRC} NAME_WE)
    add_executable(${APP_NAME} ${APP_SRC})
    
    if(TARGET spatial_core)
        target_link_libraries(${APP_NAME} PRIVATE spatial_core)
    else()
        target_include_directories(${APP_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
        )
        target_link_libraries(${APP_NAME} PRIVATE ${OpenCV_LIBS})
    endif()
    
    set_target_properties(${APP_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    message(STATUS "Configured app executable: ${APP_NAME} from ${APP_SRC}")
endforeach()

if(NOT SPATIAL_APP_SOURCES)
    message(STATUS "No files in apps/ yet. Create your main.cpp in apps/ to build an executable.")
endif()
