# -----------------------------------------------------------------------------
# Set sources and create executable target
# -----------------------------------------------------------------------------
FILE(GLOB HEADERS "uncertaintyTE/*.h")
FILE(GLOB SOURCES "uncertaintyTE/*.cpp")

FILE(GLOB MODUL_DEFINITION_FILES "${PROJECT_SOURCE_DIR}/src/*.def")
SET(LIBS_FILES ${LIBS_FILES}) 
SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

# -----------------------------------------------------------------------------
# Library output
# -----------------------------------------------------------------------------
OPTION(UNC_BUILD_STATIC_LIB "Build uncertainty as static library" OFF)
OPTION(UNC_BUILD_SHARED_LIB "Build uncertainty as shared library" ON)

IF (UNC_BUILD_STATIC_LIB)
	ADD_LIBRARY(uncertainty_static STATIC ${HEADERS} ${SOURCES})
	target_include_directories(uncertainty_static PUBLIC ${INCLUDE_DIRS})
	target_link_libraries (uncertainty_static ${LIBS_FILES})
        install(TARGETS uncertainty_static
          DESTINATION lib
          EXPORT uncertainty-targets
        )
ENDIF (UNC_BUILD_STATIC_LIB)

IF (UNC_BUILD_SHARED_LIB)
	ADD_LIBRARY(uncertainty_shared SHARED ${HEADERS} ${SOURCES})
	target_include_directories(uncertainty_shared PUBLIC ${INCLUDE_DIRS})
	target_link_libraries (uncertainty_shared ${LIBS_FILES})
        install(TARGETS uncertainty_shared
          DESTINATION lib
          EXPORT uncertainty-targets
        )
ENDIF (UNC_BUILD_SHARED_LIB)

# -----------------------------------------------------------------------------
# Executable - binary file with txt input
# -----------------------------------------------------------------------------
ADD_EXECUTABLE(uncertainty software/main_uncertainty.cpp)
target_include_directories(uncertainty
    PUBLIC
    .
    ${INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIR})

target_link_libraries(uncertainty
    ${LIBS_FILES}
    ${GFLAGS_LIBRARIES})

if(UNC_BUILD_STATIC_LIB)
    target_link_libraries (uncertainty uncertainty_static)
else()
    target_link_libraries (uncertainty uncertainty_shared)
endif()


install(DIRECTORY uncertaintyTE
  DESTINATION include
  COMPONENT headers
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
  )



# -----------------------------------------------------------------------------
# Compile MEX
# -----------------------------------------------------------------------------
OPTION(UNC_BUILD_MEX "Build uncertainty_mex file" OFF)

# Check if the matlab libs are setup
IF(UNC_BUILD_MEX AND NOT UNC_MATLAB_SUPPORT)
	MESSAGE( FATAL_ERROR "Please, include matlab libraries and headers (i.e. UNC_MATLAB_SUPPORT = ON) to build mex file (i.e. UNC_BUILD_MEX = ON)." )
ENDIF(UNC_BUILD_MEX AND NOT UNC_MATLAB_SUPPORT)

IF(UNC_BUILD_MEX AND UNC_MATLAB_SUPPORT) 
    IF(WIN32)
        add_library(uncertainty_mex SHARED ${HEADERS} ${SOURCES} "mex/uncertatinty_mex.cpp" ${MODUL_DEFINITION_FILES})
		target_include_directories(uncertainty_mex
			PUBLIC
			.
			${INCLUDE_DIRS}
			${GFLAGS_INCLUDE_DIR})

		target_link_libraries(uncertainty_mex
			${LIBS_FILES}
			${GFLAGS_LIBRARIES})
		
		#if(UNC_BUILD_STATIC_LIB)
		#	target_link_libraries (uncertainty_mex uncertainty_static)
		#else()
		#	target_link_libraries (uncertainty_mex uncertainty_shared)
		#endif()

        #target_include_directories(uncertainty_mex PUBLIC ${INCLUDE_DIRS})
        #target_link_libraries (uncertainty_mex ${LIBS_FILES})
        add_definitions(-DMATLAB_MEX_FILE)
        SET_TARGET_PROPERTIES(uncertainty_mex PROPERTIES LINK_FLAGS "/export:mexFunction /dll")
        SET_TARGET_PROPERTIES(uncertainty_mex PROPERTIES SUFFIX ".mexw64")
			
	ELSE(WIN32)
            # Build shared library, create another project, generate make, execute it
            ADD_LIBRARY(unc STATIC ${HEADERS} ${SOURCES})
            target_include_directories(unc PUBLIC ${INCLUDE_DIRS})
            target_link_libraries (unc ${LIBS_FILES})
            set_property(TARGET unc PROPERTY POSITION_INDEPENDENT_CODE ON)

            # Generate shell script for compilation
            SET(CM_DIR "${PROJECT_SOURCE_DIR}/mex")
            SET(MEX_SH "${CM_DIR}/mex_link.sh")
            file(WRITE ${MEX_SH}
                "g++ -g -Wall -fPIC -std=c++11 -pthread -fPIC -DMX_COMPAT_32 -DMATLAB_MEX_FILE -DUSE_MATLAB "
                "-I${Matlab_INCLUDE_DIRS} -I../include -c ../mex/uncertainty_mex.cpp -o ../mex/uncertainty_mex.o \n")
            file(APPEND ${MEX_SH} 
                "mex -g -cxx CC='/usr/bin/g++' CXX='/usr/bin/g++' LD='/usr/bin/g++' GCC='/usr/bin/g++' "
                "-DUSE_MATLAB -L./ -L/media/xxxxxx/DATA/xxxxxx/libs/magma-2.2.0/build/lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib -L/usr/local/lib "
                "-lpthread -lunc -lmagma -llapacktest -ltester -lcudart_static -lrt -lcublas -lcusparse "
                "-llapack -lf77blas -latlas -lsuitesparseconfig -lspqr -lcholmod -lccolamd -lcamd -lcolamd "
                "-lcolamd -lamd -lgflags -lceres -ldl -lrt -largeArrayDims "
                "LDFLAGS='$LDFLAGS -fopenmp' -output ../mex/uncertainty_mex ../mex/uncertainty_mex.o\n") 
            file(APPEND ${MEX_SH} 
                "mv ../mex/uncertainty_mex.mexa64 ${CMAKE_BINARY_DIR}")   

            # Execute the script after the build
            ADD_CUSTOM_COMMAND(TARGET unc POST_BUILD COMMAND "sh" ${MEX_SH})
        ENDIF(WIN32)
ENDIF(UNC_BUILD_MEX AND UNC_MATLAB_SUPPORT) 


# ==============================================================================
# UNIT TEST framework
# ==============================================================================
if(UNC_BUILD_TESTS)
  enable_testing()
endif()

# MACRO to ease UnitTesting
macro(UNIT_TEST NAMESPACE NAME EXTRA_LIBS)
  if(UNC_BUILD_TESTS)
    add_executable(${NAMESPACE}_test_${NAME} ${NAME}_test.cpp)

    set_property(TARGET ${NAMESPACE}_test_${NAME} PROPERTY FOLDER uncertaintyTE/test)

    target_link_libraries(${NAMESPACE}_test_${NAME}
                          ${EXTRA_LIBS} # Extra libs MUST be first.
                          ${BOOST_LIBRARIES} uncertainty_shared)
	
	target_include_directories(${NAMESPACE}_test_${NAME}
				PUBLIC
				.
				${INCLUDE_DIRS})

    add_test(NAME ${NAMESPACE}_test_${NAME}
			 WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
			 COMMAND $<TARGET_FILE:${NAMESPACE}_test_${NAME}> --log_level=all)
  endif()
endmacro()


# Tests 
add_definitions(-DTESTS_ROOT_DIR="${CMAKE_SOURCE_DIR}/test")
add_subdirectory(test_eigValues)




