CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )

project (GGNN)
set(CMAKE_CXX_STANDARD 14)


find_package(CUDA REQUIRED)

set(CUDA_ARCH_LIST Auto)
# set(CUDA_ARCH_LIST "7.0")
# set(CUDA_ARCH_LIST "7.5 6.1 8.6")
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

set(CUB_INCLUDE_PATH "3rdparty/cub")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -Xptxas=-v --expt-relaxed-constexpr --default-stream per-thread" )
# debug
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
# profile
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")

add_subdirectory("3rdparty/pybind11")
set(GFLAGS_NAMESPACE "google")
add_subdirectory("3rdparty/glog")

message("CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

# configurations for specific datasets
# TODO: generate from template using CMake?

set(KBUILD_VALUES 24 40) # 24 40
set(SEG_VALUES 32) # 32 64
set(KQUERY_VALUES 10) # 10 20 40
set(MAXITER_VALUES 200) # 200 400 800

foreach(KBUILD ${KBUILD_VALUES})
  foreach(SEG ${SEG_VALUES})
    foreach(KQUERY ${KQUERY_VALUES})
      foreach(MAXITER ${MAXITER_VALUES})
        # SIFT
        cuda_add_executable(sift_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER} src/sift.cu)
        target_include_directories(sift_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER}
          PRIVATE
          ${CUDA_NVCC_FLAGS}
          ${CUB_INCLUDE_PATH}
          ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
          # add parameters and SIFT flag
          # flag is for the kernel to know which load function to use
        target_compile_definitions(sift_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER} PRIVATE KBUILD=${KBUILD} SEG=${SEG} KQUERY=${KQUERY})
        target_compile_definitions(sift_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER} PRIVATE SIFT)
        target_link_libraries(sift_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER} ${CUDA_curand_LIBRARY} glog::glog)
        
        # Deep
        cuda_add_executable(deep_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER} src/deep.cu)
        target_include_directories(deep_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER}
          PRIVATE
          ${CUDA_NVCC_FLAGS}
          ${CUB_INCLUDE_PATH}
          ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_compile_definitions(deep_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER} PRIVATE KBUILD=${KBUILD} SEG=${SEG} KQUERY=${KQUERY})
        target_compile_definitions(deep_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER} PRIVATE DEEP)
        target_link_libraries(deep_KB${KBUILD}_S${SEG}_KQ${KQUERY}_MI${MAXITER} ${CUDA_curand_LIBRARY} glog::glog) 
      endforeach()
    endforeach()
  endforeach()
endforeach()

cuda_add_executable(sift1m src/sift1m.cu)
target_include_directories(sift1m
  PRIVATE
  ${CUDA_NVCC_FLAGS}
  ${CUB_INCLUDE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(sift1m ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_multi src/sift1m_multi_gpu.cu)
# target_include_directories(sift1m_multi
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_multi ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1b_subsets src/sift1b_subsets.cu)
# target_include_directories(sift1b_subsets
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1b_subsets ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1b_multi src/sift1b_multi_gpu.cu)
# target_include_directories(sift1b_multi
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1b_multi ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1b_multi_top10 src/sift1b_multi_gpu_top10.cu)
# target_include_directories(sift1b_multi_top10
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1b_multi_top10 ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(deep1b_multi src/deep1b_multi_gpu.cu)
# target_include_directories(deep1b_multi
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(deep1b_multi ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_stats src/sift1m_stats.cu)
# target_include_directories(sift1m_stats
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_stats ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_no_slack_query src/sift1m_no_slack_query.cu)
# target_include_directories(sift1m_no_slack_query
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_no_slack_query ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_subsets src/sift1m_subsets.cu)
# target_include_directories(sift1m_subsets
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_subsets ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_batchsize src/sift1m_batchsize.cu)
# target_include_directories(sift1m_batchsize
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_batchsize ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_base_vs_base src/sift1m_base_vs_base.cu)
# target_include_directories(sift1m_base_vs_base
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_base_vs_base ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_load_hnsw src/sift1m_load_hnsw.cu)
# target_include_directories(sift1m_load_hnsw
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_load_hnsw ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_top10 src/sift1m_top10.cu)
# target_include_directories(sift1m_top10
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_top10 ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(sift1m_top100 src/sift1m_top100.cu)
# target_include_directories(sift1m_top100
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(sift1m_top100 ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(nytimes src/nytimes.cu)
# target_include_directories(nytimes
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(nytimes ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(nytimes_top10 src/nytimes_top10.cu)
# target_include_directories(nytimes_top10
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(nytimes_top10 ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(glove200 src/glove200.cu)
# target_include_directories(glove200
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(glove200 ${CUDA_curand_LIBRARY} glog::glog)

# cuda_add_executable(gist src/gist.cu)
# target_include_directories(gist
#   PRIVATE
#   ${CUDA_NVCC_FLAGS}
#   ${CUB_INCLUDE_PATH}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_link_libraries(gist ${CUDA_curand_LIBRARY} glog::glog)
