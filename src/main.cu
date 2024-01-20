#include <chrono>
#include <iostream>
#include <vector>

#include "correlator.cuh"
#include "options.hpp"
#include "utils.hpp"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
  Options options(argc, argv);
  Correlator<uint32_t> correlator(options.num_bins + 1, options.bin_size, options.num_sensors, options.packet_size, 0, options.debug);

  if(options.debug) std::cout << "[INFO] Reading input file" << std::endl;
  std::vector<uint32_t> data = utils::parse_csv<uint32_t>(options.input_file);

  size_t total_instants = data.size() / options.num_sensors;
  size_t total_packets = total_instants / options.packet_size;

  if(options.debug) {
    std::cout << "[INFO] Total instants: " << total_instants << std::endl;
    std::cout << "[INFO] Total packets: " << total_packets << std::endl;
  }

  correlator.alloc();

  for(size_t i = 0; i < options.iterations; ++i) {
    correlator.reset();

    auto start = clock_type::now();

    if(options.debug) {
      std::cout << "[INFO] --------------------------------------" << std::endl;
      std::cout << "[INFO] Starting iteration: " << i << std::endl;
    }
    for(size_t j = 0; j < total_packets; ++j) {
      if(options.debug) std::cout << "[INFO] Executing packet " << j << std::endl;

      size_t starting_position = j * options.packet_size * options.num_sensors;
      correlator.correlate(data.data() + starting_position, options.packet_size);
    }

    auto end = clock_type::now();
    auto duration = chrono::duration<double>(end - start);

    if(!options.debug) std::cout << duration.count() << std::endl;
  }

  correlator.transfer();

  if(options.results) {
    std::ofstream output_file(options.output_file);
    auto taus = utils::generate_taus(total_instants, options.bin_size, options.num_bins + 1);

    if(options.debug) {
      std::cout << "[INFO] --------------------------------------" << std::endl;
      std::cout << "[INFO] number of bins: " << options.num_bins + 1 << " -- bin size: " << options.bin_size << std::endl;
      std::cout << "[INFO] maximum lag available: " << taus.size() << std::endl;
    }

    for(int lag = 0; lag < taus.size(); lag++) {
      output_file << taus[lag];
      for(int sensor = 0; sensor < options.num_sensors; ++sensor) {
        auto value = correlator.get(sensor, lag);
        output_file << ',' << value;
      }
      output_file << std::endl;
    }
    output_file.close();
  }

  return 0;
}