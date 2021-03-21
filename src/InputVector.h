#ifndef AUTOCORRELATIONCUDA_INPUTVECTOR
#define AUTOCORRELATIONCUDA_INPUTVECTOR

#include <vector>
#include <string>
#include "CudaInput.h"
#include "DataFile.h"

namespace AutocorrelationCUDA {

template <typename OutType>
class InputVector final : public CudaInput<OutType> {

	public:
	InputVector(std::vector<OutType> v) : base{v} {}

	InputVector(const std::string& path, const std::string& fileName, const std::string& format = ".txt") {
		AutocorrelationCUDA::DataFile<OutType> tmp{path, fileName, format};
		base = tmp.read();
	}

	std::vector<OutType> read() {
		return std::vector<OutType>{base};
	}

	std::vector<OutType> read(unsigned int valsToRead) {
		std::vector<OutType> out(valsToRead);

		for (int i = 0; i < valsToRead; ++i) {
			out[i] = base[pos%base.size()];
			pos++;
		}

		return out;
	}

	private:
	std::vector<OutType> base;
	unsigned int pos = 0;
};

}

#endif
