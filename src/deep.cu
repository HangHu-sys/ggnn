/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. Lensch
//

/*
  @Hang:
  Example command to run this file:
    ./deep_KB24_S32_KQ20 --dbname=Deep1M --graph_filename=../data/GPU_GGNN_index/Deep1M_KB24_S32_KQ10 --mode=1 --bs=2000
*/
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#include "ggnn/cuda_knn_ggnn.cuh"
#include "ggnn/utils/cuda_knn_constants.cuh"
#include "ggnn/utils/cuda_knn_dataset.cuh"
#include "ggnn/utils/cuda_knn_utils.cuh"

DEFINE_string(dbname, "", "dataset name");
DEFINE_string(graph_filename, "",
              "path to file that contains the serialized graph");
DEFINE_double(tau, 0.5, "Parameter tau");
DEFINE_int32(refinement_iterations, 2, "Number of refinement iterations");
DEFINE_int32(gpu_id, 0, "GPU id");
DEFINE_int32(mode, 0, "0: build, 1: query");
DEFINE_int32(bs, 10000, "batch size");
DEFINE_double(tau_query, 0.5, "Parameter tau for query");
// DEFINE_bool(grid_search, false,
//             "Perform queries for a wide range of parameters.");

#ifndef KBUILD
  #define KBUILD 24
#endif
#ifndef SEG
  #define SEG 32
#endif
#ifndef KQUERY
  #define KQUERY 10
#endif
#ifndef MAXITER
  #define MAXITER 400
#endif


int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  // google::LogToStderr();
  google::SetLogSymlink(google::INFO, "");

  gflags::SetUsageMessage(
      "GGNN: Graph-based GPU Nearest Neighbor Search\n"
      "by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. "
      "Lensch\n"
      "(c) 2020 Computer Graphics University of Tuebingen");
  gflags::SetVersionString("1.0.0");
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::string base_filename = "/mnt/scratch/wenqi/Faiss_experiments/deep1b/base.1B.fbin";
  std::string query_filename = "/mnt/scratch/wenqi/Faiss_experiments/deep1b/query.public.10K.fbin";
  std::string groundtruth_filename;
  size_t N_base;
  if (FLAGS_dbname == "Deep1M") {
    groundtruth_filename = "/mnt/scratch/wenqi/Faiss_experiments/deep1b/gt_idx_1M.ibin";
    N_base = 1e6;
  } else if (FLAGS_dbname == "Deep10M") {
    groundtruth_filename = "/mnt/scratch/wenqi/Faiss_experiments/deep1b/gt_idx_10M.ibin";
    N_base = 1e7;
  } else {
    LOG(FATAL) << "Unknown dataset " << FLAGS_dbname;
  }

  CHECK(file_exists(base_filename))
      << "File for base vectors has to exist";
  CHECK(file_exists(query_filename))
      << "File for perform_query vectors has to exist";

  CHECK_GE(FLAGS_tau, 0) << "Tau has to be bigger or equal 0.";
  CHECK_GE(FLAGS_refinement_iterations, 0)
      << "The number of refinement iterations has to be non-negative.";

  // ####################################################################
  // compile-time configuration
  //
  // data types
  //
  /// data type for addressing points (needs to be able to represent N)
  using KeyT = int32_t;
  /// data type of the dataset (e.g., char, int, float)
  using BaseT = float;
  /// data type of computed distances
  using ValueT = float;
  /// data type for addressing base-vectors (needs to be able to represent N*D)
  using BAddrT = uint64_t;
  /// data type for addressing the graph (needs to be able to represent
  /// N*KBuild)
  using GAddrT = uint64_t;
  //
  // dataset configuration (here: Deep1M)
  //
  /// dimension of the dataset
  const int D = 96;
  /// distance measure (Euclidean or Cosine)
  const DistanceMeasure measure = Euclidean;
  //
  // search-graph configuration
  //
  /// number of neighbors per point in the graph
  const int KBuild = KBUILD;
  /// maximum number of inverse/symmetric links (KBuild / 2 usually works best)
  const int KF = KBuild / 2;
  /// segment/batch size (needs to be > KBuild-KF)
  const int S = SEG;
  /// graph height / number of layers (4 usually performs best)
  const int L = 4;
  //
  // query configuration
  //
  /// number of neighbors to search for
  const int KQuery = KQUERY;

  assert(KBuild - KF < S);

  LOG(INFO) << "Using the following parameters " << KBuild << " (KBuild) " << KF
            << " (KF) " << S << " (S) " << L << " (L) " << D << " (D) ";

  // Set the requested GPU id, if possible.
  {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    CHECK_GE(FLAGS_gpu_id, 0) << "This GPU does not exist";
    CHECK_LT(FLAGS_gpu_id, numGpus) << "This GPU does not exist";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, FLAGS_gpu_id);
    LOG(INFO) << "device name: " << prop.name;
  }
  cudaSetDevice(FLAGS_gpu_id);


  typedef GGNN<measure, KeyT, ValueT, GAddrT, BaseT, BAddrT, D, KBuild, KF, KQuery, S> GGNN;

  GGNN m_ggnn{base_filename, query_filename,
              groundtruth_filename, L, static_cast<float>(FLAGS_tau), N_base};

  m_ggnn.ggnnMain(FLAGS_graph_filename, FLAGS_refinement_iterations);

  // Build mode stops here.
  if (FLAGS_mode == 0)
    return 0;

  auto query_function = [&m_ggnn](const float tau_query) {
    cudaMemcpyToSymbol(c_tau_query, &tau_query, sizeof(float));
    LOG(INFO) << "--";
    LOG(INFO) << "Query with tau_query " << tau_query;
    // faster for C@1 = 99%
    // LOG(INFO) << "fast query (good for C@1)";
    // m_ggnn.queryLayer<32, 200, 256, 64>();
    // better for C@10 > 99%
    LOG(INFO) << "regular query (good for C@10)";
    m_ggnn.queryLayer<32, MAXITER, 448, 64>(FLAGS_bs);
    // expensive, can get to 99.99% C@10
    // m_ggnn.queryLayer<128, 2000, 2048, 256>();
  };

  LOG(INFO) << "--";
  LOG(INFO) << "90, 95, 99% R@1, 99% C@10 (using -tau 0.5 "
                "-refinement_iterations 2):";
  query_function(FLAGS_tau_query);

  printf("done! \n");
  gflags::ShutDownCommandLineFlags();
  return 0;
}
