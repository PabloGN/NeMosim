#include <cuda_runtime.h>
#include <cutil.h>
#include <algorithm>
//#include <stdexcept>
#include <assert.h>

template<typename T>
SMatrix<T>::SMatrix(
			size_t partitionCount,
			size_t maxPartitionSize,
			size_t maxDelay,
            size_t maxSynapsesPerDelay,
			bool allocHostData,
			size_t submatrixCount) :
	m_deviceData(NULL),
	m_partitionCount(partitionCount),
	m_maxPartitionSize(maxPartitionSize),
    m_maxDelay(maxDelay),
	m_pitch(0),
	m_submatrixCount(submatrixCount)
{
	size_t width = maxSynapsesPerDelay;
	size_t height = submatrixCount * partitionCount * maxPartitionSize * maxDelay;
	size_t bytePitch = 0;
	CUDA_SAFE_CALL(
			cudaMallocPitch((void**)&m_deviceData, 
				&bytePitch, 
				width * sizeof(T),
				height);
	); 
	m_pitch = bytePitch / sizeof(T);

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	CUDA_SAFE_CALL(cudaMemset2D(m_deviceData, bytePitch, 0x0, bytePitch, height));

	//! \todo may need a default value here
	if(allocHostData) {
		m_hostData.resize(height * m_pitch);
        m_rowLength.resize(height, 0);
	}
}



template<typename T>
SMatrix<T>::~SMatrix()
{
    CUDA_SAFE_CALL(cudaFree(m_deviceData));
}



template<typename T>
size_t
SMatrix<T>::size() const
{
	return m_partitionCount * m_maxPartitionSize * m_maxDelay * m_pitch;
}



template<typename T>
T*
SMatrix<T>::deviceData() const
{
    return m_deviceData;
}



template<typename T>
size_t
SMatrix<T>::bytes() const
{
	return m_submatrixCount * size() * sizeof(T);
}



template<typename T>
size_t
SMatrix<T>::delayPitch() const
{
    return m_pitch;
}



template<typename T>
void
SMatrix<T>::copyToDevice()
{
    assert(m_submatrixCount * size() <= m_hostData.size());
    //! \todo add back exceptions
#if 0
	if(size() > m_hostData.size()) {
		throw std::logic_error("Attempt to copy Insuffient host data to device");
	}
#endif

	CUDA_SAFE_CALL(
			cudaMemcpy(
				m_deviceData,
				&m_hostData[0],
				bytes(),
				cudaMemcpyHostToDevice));
}



template<typename T>
void
SMatrix<T>::clearHostBuffer()
{
	m_hostData.clear();
}



template<typename T>
void
SMatrix<T>::moveToDevice()
{
    copyToDevice();
    clearHostBuffer();
}



template<typename T>
size_t
SMatrix<T>::offset(
        size_t sourcePartition,
		size_t sourceNeuron,
		size_t delay,
        size_t synapseIndex,
		size_t submatrix) const
{
    assert(sourcePartition < m_partitionCount);
    assert(sourceNeuron < m_maxPartitionSize);
    assert(delay <= m_maxDelay);
    assert(delay >= 1);
    assert(synapseIndex < delayPitch());
	assert(submatrix < m_submatrixCount);
    //! \todo refactor
    //! \todo have this call a method which we share with the kernel as well
    return submatrix * size()
			+ sourcePartition * m_maxPartitionSize * m_maxDelay * delayPitch()
            + sourceNeuron * m_maxDelay * delayPitch()
            + (delay-1) * delayPitch()
            + synapseIndex;
}


template<typename T>
size_t
SMatrix<T>::lenOffset(
        size_t sourcePartition,
		size_t sourceNeuron,
		size_t delay) const
{
    assert(sourcePartition < m_partitionCount);
    assert(sourceNeuron < m_maxPartitionSize);
    assert(delay <= m_maxDelay);
    assert(delay >= 1);
    //! \todo refactor
    size_t r = sourcePartition * m_maxPartitionSize * m_maxDelay 
        + sourceNeuron * m_maxDelay
        + delay - 1;
    assert(r < m_rowLength.size());
    return r;
}



template<typename T>
void
SMatrix<T>::setDelayRow(
		size_t sourcePartition,
		size_t sourceNeuron,
		size_t delay,
        const std::vector<T>& data,
		size_t submatrix)
{
	std::copy(data.begin(), data.end(), m_hostData.begin() 
            + offset(sourcePartition, sourceNeuron, delay, 0, submatrix));
    m_rowLength[lenOffset(sourcePartition, sourceNeuron, delay)] = data.size();
}



template<typename T>
size_t
SMatrix<T>::addSynapse(
        size_t sourcePartition,
        size_t sourceNeuron,
        size_t delay,
        const T& data)
{
    size_t i = lenOffset(sourcePartition, sourceNeuron, delay);
    size_t synapseIndex = m_rowLength[i];
    assert(synapseIndex < delayPitch());
    m_hostData[offset(sourcePartition, sourceNeuron, delay, synapseIndex)] = data;
    m_rowLength[i] += 1;
    return synapseIndex + 1;
}


template<typename T>
void
SMatrix<T>::fillHostBuffer(const T& val, size_t submatrix)
{
	typename std::vector<T>::iterator b = m_hostData.begin() + submatrix * size();
	std::fill(b, b + size(), val);
}
