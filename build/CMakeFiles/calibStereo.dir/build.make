# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wys/slam/camera-co-calib/calibStereo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wys/slam/camera-co-calib/calibStereo/build

# Include any dependencies generated for this target.
include CMakeFiles/calibStereo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/calibStereo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/calibStereo.dir/flags.make

CMakeFiles/calibStereo.dir/stereo_calib.cpp.o: CMakeFiles/calibStereo.dir/flags.make
CMakeFiles/calibStereo.dir/stereo_calib.cpp.o: ../stereo_calib.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wys/slam/camera-co-calib/calibStereo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/calibStereo.dir/stereo_calib.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/calibStereo.dir/stereo_calib.cpp.o -c /home/wys/slam/camera-co-calib/calibStereo/stereo_calib.cpp

CMakeFiles/calibStereo.dir/stereo_calib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/calibStereo.dir/stereo_calib.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wys/slam/camera-co-calib/calibStereo/stereo_calib.cpp > CMakeFiles/calibStereo.dir/stereo_calib.cpp.i

CMakeFiles/calibStereo.dir/stereo_calib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/calibStereo.dir/stereo_calib.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wys/slam/camera-co-calib/calibStereo/stereo_calib.cpp -o CMakeFiles/calibStereo.dir/stereo_calib.cpp.s

CMakeFiles/calibStereo.dir/stereo_calib.cpp.o.requires:

.PHONY : CMakeFiles/calibStereo.dir/stereo_calib.cpp.o.requires

CMakeFiles/calibStereo.dir/stereo_calib.cpp.o.provides: CMakeFiles/calibStereo.dir/stereo_calib.cpp.o.requires
	$(MAKE) -f CMakeFiles/calibStereo.dir/build.make CMakeFiles/calibStereo.dir/stereo_calib.cpp.o.provides.build
.PHONY : CMakeFiles/calibStereo.dir/stereo_calib.cpp.o.provides

CMakeFiles/calibStereo.dir/stereo_calib.cpp.o.provides.build: CMakeFiles/calibStereo.dir/stereo_calib.cpp.o


# Object files for target calibStereo
calibStereo_OBJECTS = \
"CMakeFiles/calibStereo.dir/stereo_calib.cpp.o"

# External object files for target calibStereo
calibStereo_EXTERNAL_OBJECTS =

calibStereo: CMakeFiles/calibStereo.dir/stereo_calib.cpp.o
calibStereo: CMakeFiles/calibStereo.dir/build.make
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_stitching.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_superres.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_videostab.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_aruco.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_bgsegm.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_bioinspired.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_ccalib.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_dpm.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_freetype.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_fuzzy.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_hdf.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_line_descriptor.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_optflow.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_reg.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_saliency.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_stereo.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_structured_light.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_surface_matching.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_tracking.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_xfeatures2d.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_ximgproc.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_xobjdetect.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_xphoto.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_shape.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_viz.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_phase_unwrapping.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_rgbd.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_calib3d.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_video.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_datasets.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_dnn.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_face.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_plot.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_text.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_features2d.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_flann.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_objdetect.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_ml.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_highgui.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_photo.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_videoio.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_imgcodecs.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_imgproc.so.3.2.0
calibStereo: /home/wys/pkg/opencv3.2/opencv-3.2.0/install/lib/libopencv_core.so.3.2.0
calibStereo: CMakeFiles/calibStereo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wys/slam/camera-co-calib/calibStereo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable calibStereo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/calibStereo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/calibStereo.dir/build: calibStereo

.PHONY : CMakeFiles/calibStereo.dir/build

CMakeFiles/calibStereo.dir/requires: CMakeFiles/calibStereo.dir/stereo_calib.cpp.o.requires

.PHONY : CMakeFiles/calibStereo.dir/requires

CMakeFiles/calibStereo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/calibStereo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/calibStereo.dir/clean

CMakeFiles/calibStereo.dir/depend:
	cd /home/wys/slam/camera-co-calib/calibStereo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wys/slam/camera-co-calib/calibStereo /home/wys/slam/camera-co-calib/calibStereo /home/wys/slam/camera-co-calib/calibStereo/build /home/wys/slam/camera-co-calib/calibStereo/build /home/wys/slam/camera-co-calib/calibStereo/build/CMakeFiles/calibStereo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/calibStereo.dir/depend

