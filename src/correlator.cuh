#pragma once

#include <cuda_runtime.h>

#include <iostream>

#define M 2  // Number of lag coalesced
namespace MultiTau {

/**
 * @brief Used to find for how many bins the coalesced new_value needs to be added,
 * returns a number between 1 and num_bins
 *
 * @param instants the current time instant
 * @param num_bins the total number of bins available
 * @returns the number of bin that needs to be accessed
 */
__inline__ __device__ int
insert_until_bin(const int instants, const int num_bins);

/**
 * @brief Implementation of the multitau correlation function based on the paper
 * "Efficient on the fly calculation of time correlation functions in computer simulations"
 * adapted to be used for multiple sensors at once
 *
 *
 * @tparam T type of the array used for the correlation calculation
 * @param new_values The new values of the time series for all the sensors, __device__ array
 * @param timepoints Number of timepoint that needs to be correlated
 * @param instants_processed Number of instants already processed for the time series
 * @param shift_register Array containing the shift_register channels for all the sensors for eac bin, __device__ array
 * @param shift_positions Array containind the new insert positions for all the sensors for each bin, __device__ array
 * @param accumulators Array containing the accumulator for all the sensors for each bin, __device__ array
 * @param correlation Array in which the correlation results are stored for all the sensors, __device__ array
 * @param num_bins Number of bins available in each correlator
 * @param num_sensors Number of total sensors
 */
template <typename T>
__global__ void
correlate(T* new_values,
          const int timepoints,
          int instants_processed,
          T* shift_register,
          uint16_t* shift_positions,
          T* accumulators,
          T* correlation,
          const int num_bins,
          const int num_sensors);

}  // namespace MultiTau

template <typename T>
class Correlator {
 public:
  Correlator(const int num_bins, const int bin_size, const int num_sensors, const int packet_size, const int device = 0, const bool debug = false);
  ~Correlator();

  /**
   * @brief Allocate all the necessary memory used by the device, and set all the arrays to 0.
   *
   */
  void alloc();

  /**
   * @brief Calculates the correlation for the new values.
   *
   * @param new_values array containing the new values of the time series, this is a __host__ allocated array
   * @param timepoints length of the new values
   */
  void correlate(const T* new_values, const int timepoints);

  /**
   * @brief Copy the correlation results from the __device__ array to the __host__ array, can be called after each correlate call and needs to be called before using the get function.
   *
   */
  void transfer();

  /**
   * @brief Returns the correlation results for a sensor and a specific lag.
   *
   *
   * @param sensor The desired sensor
   * @param lag The desired lag, a correct lag number goes from 0..num_taus
   * @returns The correlation results for the desired couple of sensor, lag
   */
  T get(const int sensor, const int lag);

  /**
   * @brief Reset all __device__ and __host__ arrays to zero,
   * useful if the same correlator needs to be used for different time series.
   */
  void reset();

 private:
  dim3 number_of_blocks;
  dim3 threads_per_block;

  /** Number of Bins used for each correlator.*/
  int num_bins;

  /** Number of channels for each correlator.*/
  int bin_size;

  /** Number of total sensors.*/
  int num_sensors;

  /** Number of sensors for which the autocorrelation is calculated for each cuda block,
   * it is comuted automatically in order to ensure an high multiprocessor utilization and to reduce shared memory usage.
   * */
  int num_sensors_per_block;

  /** Number of timepoints that are correlated for each kernel call,
   * if this number is equal to the time series length the correlation up to lag timeseries-1 is calculated,
   * if it is smaller multiple kernel call needs to be done to compute the correlations for the whole length.
   * */
  int packet_size;

  /** Amount of shared memory used per block, it is computed automatically.*/
  int shared_memory_per_block;

  /** Maximum tau that can be calculated for the current configuration.*/
  size_t max_tau;

  /** Total number of lags that are calculated.*/
  size_t num_taus;

  /** Total number of instants already processed for the current time series.*/
  int instants_processed = 0;

  bool debug;
  bool transferred;

  /** __host__ array containing the results of the correlation, holds the same memory layout of the respective __device__ array.*/
  T* correlation = nullptr;
  int* taus = nullptr;

  // DEVICE VARIABLES

  /** __device__ array containing the shift registers of the correlator for all the sensors.
   * The length of the array is num_sensors * num_bins * bin_size. The memory layout is studied to avoid bank conflicts an example is shown below.
   *
   *
   * Assumed 8 sensors per block, 10 bins, bin size 32, the memory layout will look like
   *
   *   +-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+
   *   |  s.0 b.0  |  s.0 b.0  |  ...      |  s.0 b.0  ||  s.1 b.0  |  s.1 b.0  |  ...      |  s.1 b.0  ||  ...      ||  s.7 b.0  |  s.7 b.0  |  ...      |  s.7 b.0  |
   *   |  pos 0    |  pos 2    |           |  pos 31   ||  pos 0    |  pos 2    |           |  pos 31   ||  ...      ||  pos 0    |  pos 2    |           |  pos 31   |
   *   +-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+
   *   |  s.0 b.1  |  s.0 b.1  |  ...      |  s.0 b.1  ||  s.1 b.1  |  s.1 b.1  |  ...      |  s.1 b.1  ||  ...      ||  s.7 b.1  |  s.7 b.1  |  ...      |  s.7 b.1  |
   *   |  pos 0    |  pos 2    |           |  pos 31   ||  pos 0    |  pos 2    |           |  pos 31   ||  ...      ||  pos 0    |  pos 2    |           |  pos 31   |
   *   +-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+
   *   |  ...      |  ...      |  ...      |  ...      ||  ...      |  ...      |  ...      |  ...      ||  ...      ||  ...      |  ...      |  ...      |  ...      |
   *   |           |           |           |           ||           |           |           |           ||  ...      ||           |           |           |           |
   *   +-----------+-----------+-----------+-----------++-----------+-----------+-----------+-----------++-----------++-----------+-----------+-----------+-----------+
   *   |  s.0 b.9  |  s.0 b.9  |  ...      |  s.0 b.9  ||  s.1 b.9  |  s.1 b.9  |  ...      |  s.1 b.9  ||  ...      ||  s.7 b.9  |  s.7 b.9  |  ...      |  s.7 b.9  |
   *   |  pos 0    |  pos 2    |           |  pos 31   ||  pos 0    |  pos 2    |           |  pos 31   ||  ...      ||  pos 0    |  pos 2    |           |  pos 31   |
   *   +===========+===========+===========+===========++===========+===========+===========+===========++===========++===========+===========+===========+===========+
   *
   * */
  T* d_shift_register = nullptr;

  /**
   * __device__ array containng the correlation results for all the sensors.
   * Has the same length and same layout of the d_shift_register array for banck conflict avoidance and faster memory reads and writes.
   *
   */
  T* d_correlation = nullptr;

  /**
   * __device__ array that contains the insert index for the new values  for the shift_register of each bins.
   * This is used to achieve a faster shift operation using a round robin approach. The length of the array is num_bins * num_sensors_per_block.
   * The memory layout is studied to avoid bank conflicts and achieve faster memory accesses. An example is show below.
   *
   *
   * Assumed 8 sensors per block, 10 bins

   *		+-----------+-----------+-----------+-----------+
   *		| shift pos | shift pos |   ...     | shift pos |
   *		|  s.0 b.0  |  s.1 b.0  |           |  s.7 b.0  |
   *		+-----------+-----------+-----------+-----------+
   *		| shift pos | shift pos |   ...     | shift pos |
   *		|  s.0 b.1  |  s.1 b.1  |           |  s.7 b.1  |
   *		+-----------+-----------+-----------+-----------+
   *		|   ...     |   ...     |   ...     |   ...     |
   *		|           |           |           |           |
   *		+-----------+-----------+-----------+-----------+
   *		| shift pos | shift pos |   ...     | shift pos |
   *		|  s.0 b.9  |  s.1 b.9  |           |  s.7 b.9  |
   *		+===========+===========+===========+===========+
   *		| shift pos | shift pos |   ...     | shift pos |
   *		|  s.8 b.0  |  s.9 b.0  |           |  s.15 b.  |
   *		+-----------+-----------+-----------+-----------+
   *		| shift pos | shift pos |   ...     | shift pos |
   *		|  s.8 b.1  |  s.9 b.1  |           |  s.15 b.  |
   *		+-----------+-----------+-----------+-----------+
   *		|   ...     |   ...     |   ...     |   ...     |
   *		|           |           |           |           |
   *		+-----------+-----------+-----------+-----------+
   *		| shift pos | shift pos |   ...     | shift pos |
   *		|  s.8 b.9  |  s.9 b.9  |           |  s.15 b.  |
   *		+===========+===========+===========+===========+
   *
   */
  uint16_t* d_shift_positions = nullptr;

  /**
   * __device__ array that contains the accumulators of each bin for all the sensors.
   * The length of the array is num_bins * num_sensors_per_block, the mempory layout is similar to the one shown above for the d_shift_positions.
   *
   */
  T* d_accumulators = nullptr;

  /** __device__ arrays that contains the new values of the time series that needs to be correlated.
   */
  T* d_new_values = nullptr;
};