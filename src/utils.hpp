#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils {

template <typename T>
T convert_to(const std::string &cell) {
  std::istringstream ss(cell);
  T value;

  ss >> value;

  if(ss.fail()) {
    throw std::runtime_error("ERROR: Could not convert content to type");
  }

  return value;
}

template <typename T>
std::vector<T> parse_csv_line(const std::string &line) {
  std::vector<T> result;
  std::stringstream lineStream(line);
  std::string cell;

  while(std::getline(lineStream, cell, ',')) {
    result.push_back(convert_to<T>(cell));
  }

  return result;
}

template <typename T>
std::vector<T> parse_csv(const std::string &filepath) {
  std::vector<T> result;
  std::ifstream file(filepath);

  if(!file.is_open()) throw std::runtime_error("ERROR: Could not open file" + filepath);

  // Skip header line
  std::string line;
  std::getline(file, line);

  // Read data into result array
  while(std::getline(file, line)) {
    std::vector<T> lineData = parse_csv_line<T>(line);
    result.insert(result.end(), lineData.begin(), lineData.end());
  }

  file.close();
  return result;
}

template <typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n) {
  auto first = v.cbegin() + m;
  auto last = v.cbegin() + n + 1;

  std::vector<T> vec(first, last);
  return vec;
}

std::vector<uint16_t> generate_taus(const size_t full_time_len, const size_t bin_size, const size_t num_bins, const size_t m = 2) {
  std::vector<uint16_t> taus;
  // size_t max_lag = std::pow((double)m, num_bins - 1) * (bin_size/2) + bin_size;
  size_t max_lag = std::pow((double)m, num_bins - 1) * (bin_size - 1);

  for(size_t i = 0; i < bin_size; ++i) {
    if(i > full_time_len) {
      return taus;
    }
    taus.push_back(i);
  }

  for(size_t i = 1; i < num_bins; ++i) {
    for(size_t j = (bin_size / 2); j < bin_size; ++j) {
      size_t p = std::pow((double)m, i);
      size_t tau = j * p;
      if(tau > max_lag || tau > full_time_len - p) {
        return taus;
      }
      taus.push_back(tau);
    }
  }

  return taus;
}
}  // namespace utils