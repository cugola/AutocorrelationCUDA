#pragma once

#include <getopt.h>

#include <iostream>
#include <string>

#define DEBUG false
#define ITERATIONS 1
#define PARSE_FILE false
#define NUM_BINS 10
#define BIN_SIZE 32
#define NUM_SENSORS 1024

struct Options {
  bool debug = DEBUG;

  uint iterations = ITERATIONS;
  int packet_size = -1;
  size_t num_bins = NUM_BINS;
  size_t bin_size = BIN_SIZE;
  size_t num_sensors = NUM_SENSORS;

  bool results = false;
  bool parse_file = PARSE_FILE;

  std::string input_file;
  std::string output_file;

  Options(int argc, char* argv[]) {
    int opt;
    int option_index = 0;
    static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                           {"results", no_argument, 0, 'r'},
                                           {"packets", required_argument, 0, 'p'},
                                           {"input_file", required_argument, 0, 'i'},
                                           {"output_file", optional_argument, 0, 'o'},
                                           {"iterations", optional_argument, 0, 'I'},
                                           {"sensors", optional_argument, 0, 's'},
                                           {"levels", optional_argument, 0, 'l'},
                                           {"group_size", optional_argument, 0, 'g'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    while((opt = getopt_long(argc, argv, "hdrp:i:o:I:s:l:g:", long_options, &option_index)) != EOF) {
      switch(opt) {
        case 'd':
          debug = true;
          break;
        case 'r':
          results = true;
          break;
        case 'p':
          packet_size = atoi(optarg);
          break;
        case 'i':
          input_file = optarg;
          parse_file = true;
          break;
        case 'o':
          output_file = optarg;
          break;
        case 'I':
          iterations = atoi(optarg);
          break;
        case 's':
          num_sensors = atoi(optarg);
          break;
        case 'l':
          num_bins = atoi(optarg);
          break;
        case 'g':
          bin_size = atoi(optarg);
          break;
        case 'h':
          std::cout << "This program is designed to perform autocorrelation calculations on a matrix of sensor data" << std::endl;
          std::cout << "using CUDA, a parallel computing platform." << std::endl
                    << std::endl;
          std::cout << "    [--debug, -d]           Activate debug prints" << std::endl
                    << std::endl;
          std::cout << "    [--results, -r]         Prints to stdout the results of the autocorrelation" << std::endl
                    << std::endl;
          std::cout << "    [--packets, -p]         Number of instant used per packets" << std::endl
                    << std::endl;
          std::cout << "    [--input_file, -i]      Name of the input file containing the sensor data" << std::endl;
          std::cout << "                            for the calculation of the autocorrelation" << std::endl
                    << std::endl;
          std::cout << "    [--output-file, -o]     Name of the output file. Correlation result will be saved into a csv file." << std::endl;
          std::cout << "                            If not given a default name will be used \"out.csv\"" << std::endl
                    << std::endl;
          std::cout << "    [--iterations, -I]      Number of times that the calculation is repeated. If it is greater than" << std::endl;
          std::cout << "                            one the calculation of the autocorrelation will be repeated multiple" << std::endl;
          std::cout << "                            times on the same data" << std::endl
                    << std::endl;
          std::cout << "    [--sensors, -s]         Number of Sensors present in the matrix of sensors" << std::endl
                    << std::endl;
          std::cout << "    [--groups, -l]          Number of bins used for each correlator of the sensor matrix" << std::endl
                    << std::endl;
          std::cout << "    [--group_size, -g]      Size of the bins for each correlator" << std::endl
                    << std::endl;
          std::cout << "    [--help, -h]            Print this help message" << std::endl
                    << std::endl;
          exit(0);
          break;
        default:
          break;
      }
    }

    if(packet_size == -1) {
      std::cerr << "ERROR: --packets is a required argument. Use --help for more details" << std::endl;
      exit(1);
    }

    if(!parse_file) {
      std::cout << "ERROR: --input_file is a required argument. Use --help for more details" << std::endl;
      exit(1);
    }

    if(results && output_file.empty()) {
      std::cout << "ERROR: --output_file is a required argument. Use --help for more details" << std::endl;
      exit(1);
    }
  }
};