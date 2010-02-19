#include "ConnectivityMatrix.hpp"
#include "ConnectivityMatrixImpl.hpp"


ConnectivityMatrix::ConnectivityMatrix(
        size_t maxPartitionSize,
		bool setReverse) :
	m_impl(new ConnectivityMatrixImpl(maxPartitionSize, setReverse)) {}



ConnectivityMatrix::~ConnectivityMatrix()
{
	delete m_impl;
}


void
ConnectivityMatrix::setRow(
		uint sourceNeuron,
		const uint* f_targetNeuron,
		const uint* f_delays,
		const float* f_weights,
		const uchar* f_isPlastic,
		size_t length)
{
	m_impl->setRow(
		sourceNeuron,
		f_targetNeuron,
		f_delays,
		f_weights,
		f_isPlastic,
		length);
}



void
ConnectivityMatrix::moveToDevice()
{
	m_impl->moveToDevice();
}



size_t
ConnectivityMatrix::getRow(
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		uint currentCycle,
		pidx_t* partition[],
		nidx_t* neuron[],
		weight_t* weight[],
		uchar* plastic[])
{
	return m_impl->getRow(sourcePartition, sourceNeuron,
			delay, currentCycle, partition, neuron, weight, plastic);
}



void
ConnectivityMatrix::clearStdpAccumulator()
{
	m_impl->clearStdpAccumulator();
}


size_t
ConnectivityMatrix::d_allocated() const
{
	return m_impl->d_allocated();
}



delay_t
ConnectivityMatrix::maxDelay() const
{
	return m_impl->maxDelay();
}


synapse_t*
ConnectivityMatrix::d_fcm() const
{
	return m_impl->d_fcm();
}



outgoing_t*
ConnectivityMatrix::outgoing() const
{
	return m_impl->outgoing();
}



uint*
ConnectivityMatrix::outgoingCount() const
{
	return m_impl->outgoingCount();
}



incoming_t*
ConnectivityMatrix::incoming() const
{
	return m_impl->incoming();
}



uint*
ConnectivityMatrix::incomingHeads() const
{
	return m_impl->incomingHeads();
}


uint
ConnectivityMatrix::fractionalBits() const
{
	return m_impl->fractionalBits();
}