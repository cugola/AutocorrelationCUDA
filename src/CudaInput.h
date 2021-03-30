#ifndef AUTOCORRELATIONCUDA_CUDAINPUT
#define AUTOCORRELATIONCUDA_CUDAINPUT

#include <vector>

namespace AutocorrelationCUDA {

template <typename OutType>
class CudaInput {
	public:
	virtual std::vector<OutType> read() = 0;
	virtual std::vector<OutType> read(std::uint_fast32_t valsToRead) = 0;
};
}


#endif
