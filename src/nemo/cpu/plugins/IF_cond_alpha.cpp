#ifndef NEMO_CPU_PLUGINS_IF_COND_ALPHA
#define NEMO_CPU_PLUGINS_IF_COND_ALPHA

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file IF_cond_alpha.cpp Neuron update CPU kernel for conductance-based
 * alpha decaying integrate-and-fire neurons. */

#include <cassert>

#include <nemo/fixedpoint.hpp>
#include <nemo/plugins/IF_cond_alpha.h>

#include "neuron_model.h"


extern "C"
NEMO_PLUGIN_DLL_PUBLIC
void
cpu_update_neurons(
		unsigned start, unsigned end,
		unsigned cycle,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateHistoryStride, size_t stateVarStride,
		unsigned fbits,
		unsigned fstim[],
		RNG rng[],
		float currentEPSP[],
		float currentIPSP[],
		float currentExternal[],
		uint64_t recentFiring[],
		unsigned fired[],
		void* /* rcm */)
{
	const float* p_v_rest     = paramBase + PARAM_V_REST     * paramStride;
	const float* p_c_m        = paramBase + PARAM_C_M        * paramStride;
	const float* p_tau_m      = paramBase + PARAM_TAU_M      * paramStride;
	const float* p_tau_refrac = paramBase + PARAM_TAU_REFRAC * paramStride;
	const float* p_tau_syn_E  = paramBase + PARAM_TAU_SYN_E  * paramStride;
	const float* p_tau_syn_I  = paramBase + PARAM_TAU_SYN_I  * paramStride;
	const float* p_I_offset   = paramBase + PARAM_I_OFFSET   * paramStride;
	const float* p_v_reset    = paramBase + PARAM_V_RESET    * paramStride;
	const float* p_v_thresh   = paramBase + PARAM_V_THRESH   * paramStride;
    const float* p_E_rev      = paramBase + PARAM_E_REV      * paramStride;
    const float* p_I_rev      = paramBase + PARAM_I_REV      * paramStride;

	const size_t historyLength = 1;

	/* Current state */
	size_t b0 = cycle % historyLength;
	const float* v0  = stateBase + b0 * stateHistoryStride + STATE_V * stateVarStride;
	const float* Ie0 = stateBase + b0 * stateHistoryStride + STATE_IE * stateVarStride;
	const float* Ii0 = stateBase + b0 * stateHistoryStride + STATE_II * stateVarStride;
	const float* DIe0 = stateBase + b0 * stateHistoryStride + STATE_DIE * stateVarStride;
	const float* DIi0 = stateBase + b0 * stateHistoryStride + STATE_DII * stateVarStride;
	const float* lastfired0 = stateBase + b0 * stateHistoryStride + STATE_LASTFIRED * stateVarStride;

	/* Next state */
	size_t b1 = (cycle+1) % historyLength;
	float* v1  = stateBase + b1 * stateHistoryStride + STATE_V * stateVarStride;
	float* Ie1 = stateBase + b1 * stateHistoryStride + STATE_IE * stateVarStride;
	float* Ii1 = stateBase + b1 * stateHistoryStride + STATE_II * stateVarStride;
	float* DIe1 = stateBase + b1 * stateHistoryStride + STATE_DIE * stateVarStride;
	float* DIi1 = stateBase + b1 * stateHistoryStride + STATE_DII * stateVarStride;
	float* lastfired1 = stateBase + b1 * stateHistoryStride + STATE_LASTFIRED * stateVarStride;

	/* Each neuron has two indices: a local index (within the group containing
	 * neurons of the same type) and a global index. */

	int nn = end-start;
	assert(nn >= 0);

#pragma omp parallel for default(shared)
	for(int nl=0; nl < nn; nl++) {

		unsigned ng = start + nl;

		//! \todo consider pre-multiplying tau_syn_E/tau_syn_I
		//! \todo use euler method for the decay as well?
    
        DIe1[nl] = (1.0f - 1.0f/p_tau_syn_E[nl]) * DIe0[nl] + currentEPSP[ng]; 
        DIi1[nl] = (1.0f - 1.0f/p_tau_syn_I[nl]) * DIi0[nl] + currentIPSP[ng];
		float Ie = Ie0[nl] + (2.7182818284590451f * DIe1[nl] - Ie0[nl])/p_tau_syn_E[nl];
		float Ii = Ii0[nl] + (2.7182818284590451f * DIi1[nl] - Ii0[nl])/p_tau_syn_I[nl];

        float v = v0[nl];

		/* Update the incoming current */
		float I = Ie * (p_E_rev[nl] - v) - Ii * (p_I_rev[nl] - v) + currentExternal[ng] + p_I_offset[nl];
		Ie1[nl] = Ie;
		Ii1[nl] = Ii;

		/* no need to clear current?PSP. */

		//! \todo clear this outside kernel. Make sure to do this in all kernels.
		currentExternal[ng] = 0.0f;

		
		bool refractory = lastfired0[nl] <= p_tau_refrac[nl];

		/* If we're in the refractory period, no internal dynamics */
		if(!refractory) {
			float c_m = p_c_m[nl];
			float v_rest = p_v_rest[nl];
			float tau_m = p_tau_m[nl];
			//! \todo make integration step size a model parameter as well
			v += I / c_m + (v_rest - v) / tau_m;
		}

		/* Firing can be forced externally, even during refractory period */
		fired[ng] = (!refractory && v > p_v_thresh[nl]) || fstim[ng] ;
		fstim[ng] = 0;
		recentFiring[ng] = (recentFiring[ng] << 1) | (uint64_t) fired[ng];

		if(fired[ng]) {
			// reset refractory counter
			//! \todo make this a built-in integer type instead
			lastfired1[nl] = 1;
			v1[nl] = p_v_reset[nl];
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		} else {
			lastfired1[nl] += 1;
			v1[nl] = v;
		}
	}
}


cpu_update_neurons_t* test = &cpu_update_neurons;


#include "default_init.c"

#endif[nl]
