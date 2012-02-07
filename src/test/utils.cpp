#include <cmath>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>

#include <nemo/izhikevich.hpp>
#include "utils.hpp"

void
runSimulation(
		const nemo::Network* net,
		nemo::Configuration conf,
		unsigned seconds,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx,
		bool stdp,
		std::vector<unsigned> initFstim,
		std::vector< std::pair<unsigned, float> > initIstim)
{
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	fcycles->clear();
	fnidx->clear();

	std::vector<unsigned> noFstim;
	std::vector< std::pair<unsigned, float> > noIstim;

	//! todo vary the step size between reads to firing buffer
	
	for(unsigned s = 0; s < seconds; ++s) {
		for(unsigned ms = 0; ms < 1000; ++ms) {

			bool stimulate = s == 0 && ms == 0;
			const std::vector<unsigned>& fired =
					sim->step(stimulate ? initFstim : noFstim,
							stimulate ? initIstim : noIstim);

			std::copy(fired.begin(), fired.end(), back_inserter(*fnidx));
			std::fill_n(back_inserter(*fcycles), fired.size(), s*1000 + ms);
		}
		if(stdp) {
			sim->applyStdp(1.0);
		}
	}

	BOOST_CHECK_EQUAL(fcycles->size(), fnidx->size());
}



void
compareSimulationResults(
		const std::vector<unsigned>& cycles1,
		const std::vector<unsigned>& nidx1,
		const std::vector<unsigned>& cycles2,
		const std::vector<unsigned>& nidx2)
{
	BOOST_CHECK_EQUAL(nidx1.size(), nidx2.size());
	BOOST_CHECK_EQUAL(cycles1.size(), cycles2.size());

	for(size_t i = 0; i < cycles1.size(); ++i) {
		// no point continuing after first divergence, it's only going to make
		// output hard to read.
		BOOST_CHECK_EQUAL(cycles1.at(i), cycles2.at(i));
		BOOST_CHECK_EQUAL(nidx1.at(i), nidx2.at(i));
		if(nidx1.at(i) != nidx2.at(i)) {
			BOOST_FAIL("c" << cycles1.at(i) << "/" << cycles2.at(i));
		}
	}
}


void
setBackend(backend_t backend, nemo::Configuration& conf)
{
	switch(backend) {
		case NEMO_BACKEND_CPU: conf.setCpuBackend(); break;
		case NEMO_BACKEND_CUDA: conf.setCudaBackend(); break;
		default: BOOST_REQUIRE(false);
	}
}


nemo::Configuration
configuration(bool stdp, unsigned partitionSize, backend_t backend)
{
	nemo::Configuration conf;

	if(stdp) {
		std::vector<float> pre(20);
		std::vector<float> post(20);
		for(unsigned i = 0; i < 20; ++i) {
			float dt = float(i + 1);
			pre.at(i) = 0.1f * expf(-dt / 20.0f);
			post.at(i) = -0.08f * expf(-dt / 20.0f);
		}
		conf.setStdpFunction(pre, post, -10.0, 10.0);
	}

	setBackend(backend, conf);
	conf.setCudaPartitionSize(partitionSize);

	return conf;
}


void
addExcitatoryNeuron(unsigned nidx, nemo::Network& net, float sigma)
{
	float r = 0.5;
	float b = 0.25f - 0.05f * r;
	float v = -65.0;
	float args[7] = {0.02f + 0.08f * r, b, v, 2.0f, sigma, b*v, v};
	unsigned ntype = 0U;
	net.addNeuron(ntype, nidx, 7, args);
}



/*! \return
 * 		neuron index of the nth neuron in a ring network with \a ncount neurons
 * 		with neuron indices starting from \a n0 and with \a nstep indices
 * 		between neighbouring neurons.
 *
 * The \a nth parameter is considered modulo ncount.
 * */
unsigned
ringNeuronIndex(unsigned nth, unsigned ncount, unsigned n0, unsigned nstep)
{
	return n0 + ((nth % ncount) * nstep);
}



void
createRing(nemo::Network* net,
		unsigned neuronType,
		unsigned synapseType,
		unsigned ncount,
		unsigned n0,
		unsigned nstep,
		unsigned delay)
{
	for(unsigned i_source=0; i_source < ncount; ++i_source) {
		float v = -65.0f;
		float b = 0.2f;
		float r = 0.5f;
		float r2 = r * r;
		unsigned source = ringNeuronIndex(i_source, ncount, n0, nstep);
		float args[7] = {0.02f, b, v+15.0f*r2, 8.0f-6.0f*r2, 0.0f, b*v, v};
		net->addNeuron(neuronType, source, 7, args);
		unsigned target = ringNeuronIndex(i_source+1, ncount, n0, nstep);
		net->addSynapse(synapseType, source, target, delay, 1000.0f);
	}
}



nemo::Network*
createRing(unsigned ncount, unsigned n0, bool plastic, unsigned nstep, unsigned delay)
{
	if(plastic) {
		throw nemo::exception(NEMO_API_UNSUPPORTED, "This version of NeMo does not support STDP");
	}

	nemo::izhikevich::Network* net = new nemo::izhikevich::Network;
	createRing(static_cast<nemo::Network*>(net), 0U, 0U, ncount, n0, nstep, delay);
	return net;
}
