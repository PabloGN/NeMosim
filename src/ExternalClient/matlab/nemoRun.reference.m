function fired = nemoRun(nsteps, fstim)

	% TODO: remove debugging code
	% TODO: use globals for file handles
	%fid = fopen('firing.dat','wt');
	fid = 1;

	global NEMO_NEURONS_A;
	global NEMO_STDP_ENABLED;
	global NEMO_CYCLE;

	fired = [];
	verbose = false;
	ncount = size(NEMO_NEURONS_A,1);

	for t=1:nsteps
		fs = firing_stimulus(fstim, t);
		I = deliver_spikes(NEMO_CYCLE, ncount);
		f_new = update_state(verbose, fid, t-1, NEMO_CYCLE, fs, I);
		if NEMO_STDP_ENABLED
			accumulateStdpStatistics(verbose, fid, NEMO_CYCLE);
		end;
		fired = [fired; f_new];
		NEMO_CYCLE = NEMO_CYCLE + 1;
	end;

	%fclose(fid);
end



% The user specifies stimulus for a whole block of cycles. Return the once
% relevant to *this* cycle.
function stim = firing_stimulus(stimulus, t)
	stim = [];
	if ~isempty(stimulus)
		stim = stimulus(stimulus(:,1) == t, 2);
	end;
end



% Deliver all spikes due for arrival this cycle, and return vector of
% accumulated current from incoming spikes for each neuron
function I = deliver_spikes(t, ncount)

	global NEMO_RECENT_FIRING;
	global NEMO_CM;
	global NEMO_MAX_DELAY;

	I = zeros(ncount, 1);
	history_len = size(NEMO_RECENT_FIRING, 2);

	for delay=1:NEMO_MAX_DELAY
		weights = NEMO_CM{delay};
		pres = NEMO_RECENT_FIRING(:, rf_idx(t-delay, history_len));
		I_acc = weights(:,pres);
		I = I + sum(I_acc,2);
	end;
end



% Update the state of each neuron
function firings_out = update_state(verbose, fid, t0, t, stimulated, I)

	global NEMO_NEURONS_A; a = NEMO_NEURONS_A;
	global NEMO_NEURONS_B; b = NEMO_NEURONS_B;
	global NEMO_NEURONS_C; c = NEMO_NEURONS_C;
	global NEMO_NEURONS_D; d = NEMO_NEURONS_D;

	global NEMO_NEURONS_U;
	global NEMO_NEURONS_V;

	% Update v and u using Izhikevich's model in increments of tau
	emsteps = 4;     % euler method steps
	tau = 1/emsteps; % ms per euler method step

	for k=1:emsteps
		quiet = find(NEMO_NEURONS_V < 30);
		NEMO_NEURONS_V(quiet) = NEMO_NEURONS_V(quiet) + ...
			tau*((0.04*NEMO_NEURONS_V(quiet)+5) .* NEMO_NEURONS_V(quiet) ...
				+ 140 - NEMO_NEURONS_U(quiet) + I(quiet));
		NEMO_NEURONS_U(quiet) = NEMO_NEURONS_U(quiet) + ...
			tau*(a(quiet).*(b(quiet).*NEMO_NEURONS_V(quiet)-NEMO_NEURONS_U(quiet)));
	end;

	fired = find(NEMO_NEURONS_V >= 30);
	fired = sort([fired; stimulated]);
	if verbose && ~isempty(fired)
		global NEMO_CYCLE;
		for i=1:length(fired)
			fprintf(fid, 'c%u: n%u fired\n', NEMO_CYCLE, fired(i));
		end;
	end;

	if ~isempty(fired)
		NEMO_NEURONS_V(fired) = c(fired);
		NEMO_NEURONS_U(fired) = NEMO_NEURONS_U(fired) + d(fired);
	end;

	firings_out = [t0+1+0*fired, fired];

	global NEMO_RTS_FIRED;
	NEMO_RTS_FIRED = NEMO_RTS_FIRED + numel(fired);

	% Set recent firing history (a rotating buffer)
	global NEMO_RECENT_FIRING;
	history_len = size(NEMO_RECENT_FIRING, 2);
	now = rf_idx(t, history_len);
	NEMO_RECENT_FIRING(:, now) = 0;
	if ~isempty(fired)
		NEMO_RECENT_FIRING(fired, now) = 1;
	end;
end



function accumulateStdpStatistics(verbose, fid, t)

	global NEMO_STDP_PREFIRE;
	global NEMO_STDP_POSTFIRE;
	global NEMO_RECENT_FIRING;
	global NEMO_RCM;
	global NEMO_RCM_CHANGED;
	global NEMO_RCM_VALID;
	global NEMO_STDP_ACC;
	global NEMO_MAX_DELAY;

	global NEMO_RTS_LTP;
	global NEMO_RTS_LTD;
	global NEMO_RTS_ENABLED;

	% Get all neurons which fired in the middle of the STDP window
	tf = int32(t - length(NEMO_STDP_POSTFIRE));

	% All potentially relevent post-synaptic firings

	history_len = size(NEMO_RECENT_FIRING, 2);
	post_firings = find(NEMO_RECENT_FIRING(:, rf_idx(tf, history_len)));

	% Each column in the recent firing history corresponds to a particular
	% recent cycle. This is a circular buffer where the beginning (current
	% cycle) moves around. The t_buffer vector contains the cycle number
	% corresponding to each column in the buffer.
	t_buffer = int32(0:63) + int32(floor(t/64) * 64);
	t_buffer(t_buffer > t) = t_buffer(t_buffer > t) - 64;

	stdp_pre_window = length(NEMO_STDP_PREFIRE);
	stdp_post_window = length(NEMO_STDP_POSTFIRE);
	stdp_len = stdp_pre_window + stdp_post_window;

	% Combined lookup table for whole STDP region
	% TODO: keep a column format from the start
	stdp_fn = [NEMO_STDP_PREFIRE(end:-1:1)'; 0; NEMO_STDP_POSTFIRE'];


	% The spikes that fall into the potentiation and the depression parts of
	% the STDP window are handled separately. We classify spikes based on the
	% 'dt' values. The lookuptable (dt -> class) is precomputed here:
	% The classes are
	% 1: invalid
	% 2: potentiation
	% 3: depression

	stdp_regions_valid = ones(size(stdp_fn)); 
	stdp_regions_valid(sign(stdp_fn) == 1) = 2;
	stdp_regions_valid(sign(stdp_fn) == -1) = 3;

	stdp_regions = [ones(history_len-stdp_len, 1); ... % spikes arriving before STDP window
	                stdp_regions_valid;...             % spikes arriving inside STDP window
	                ones(NEMO_MAX_DELAY, 1)];          % spikes arrivaing after STDP window

	% offset when determining regions
	stdp_offset = history_len - stdp_len + stdp_pre_window + 1;

	% offset when doing the lookup based on (valid) dts
	stdp_fn_origin = stdp_pre_window + 1;

	% Pick out only post/delay combination for which postsynaptic neuron fired
	% and for which there are incoming synapses with the given delay.
	[posts, ds] = find(NEMO_RCM_VALID(post_firings,:));

	for i=1:length(posts)

		post = post_firings(posts(i));
		d = ds(i);

		% All recent firings of presynaptic neurons. Each row corresponds to a
		% single synapse. There may be several spikes for each synapse       
		pre0 = NEMO_RCM{post,d};

		% f_pre contains one row for each presynaptic neuron, with synapses
		% converging on postsynaptic.
		f_pre = NEMO_RECENT_FIRING(pre0,:);

		% Synapses (indices into firing history) for which we need to check dt.
		% 'synapses' are the row numbers in f_pre which have relevant firings.
		% TODO: avoid transpose here by storing firing history differently.
		synapses = (any(f_pre'));

		% It's possible that none of the relevant presynaptic neurons have
		% fired recently, in which case we give up early.
		if(~any(synapses))
			continue;
		end

		% i_pres and i_tfired are vectors which together describe all the
		% spikes which may take part in STDP. i_tfired are the bins in the
		% recent firing history each spike falls in under.
		[i_pres, i_tfired] = find(f_pre(synapses,:) ~= 0);                 

		% Since we have stored f_pre in row-major order we get incorrect
		% orientation on i_pres and i_tfired if we only have a single row.
		% Therefore, force into column format.
		% TODO: make sure this is already in column format
		i_pres = i_pres(:);

		%    firing time (pre) + delay - firing time (post)
		dt = t_buffer(i_tfired(:)) + d - tf;

		% Make negative indices 1-based rather than 0-based (but still negative)
		dt(dt <= 0) = dt(dt <= 0) - 1;

		% Split spikes into presynaptic and postsynaptic regions as described above.
		subs = stdp_regions(dt + stdp_offset);

		% now get minimum for the presynaptic neurons with active spikes
		len = length(find(synapses));
		dt_sorted = accumarray([subs, i_pres], dt, [3 len], @absmin);

		if NEMO_RTS_ENABLED
			NEMO_RTS_LTP = NEMO_RTS_LTP + length(find(dt_sorted(2,:) ~= 0));
			NEMO_RTS_LTD = NEMO_RTS_LTD + length(find(dt_sorted(3,:) ~= 0));
		end

		NEMO_RCM_CHANGED(post,d) = true;

		% The summing here combines potentiation and depression spikes for the same synapse
		NEMO_STDP_ACC{post,d}(synapses) = ...
			NEMO_STDP_ACC{post,d}(synapses) + ...
			sum(stdp_fn(dt_sorted(2:3,:)+stdp_fn_origin));

		if verbose
			logStdp(fid, 'ltp', t, post, pre0(synapses(dt_sorted(2,:))));
			logStdp(fid, 'ltd', t, post, pre0(synapses(dt_sorted(3,:))));
		end
	end
end



function y = absmin(xs)
	[~, i] = min(abs(xs));
	y = xs(i);
end



% Print potentiation or depression to stdout, if in verbose mode
function logStdp(fid, type, t, post, dt_arr)
	% TODO: use sparse format here for dt_arr
	for pre = 1:length(dt_arr)
		dt = dt_arr(pre);
		if dt >= 1
			fprintf(fid, 'c%u %s: %u->%u dt=%d\n', t, type, pre, post, dt-1);
		end;
	end;
end


function logStdp0(fid, type, t, post, dt_arr)
	% TODO: use sparse format here for dt_arr
	if ~isempty(dt_arr)
		for i = 1:size(dt_arr,1)
			pre = dt_arr(i,1);
			dt = dt_arr(i,2);
			%dt = dt_arr(pre);
			fprintf(fid, 'c%u %s: %u->%u dt=%d\n', t, type, pre, post, dt-1);
		end;
	end;
end



% Return index in range [1,bufsz] into circular firing buffer of size sz.
function idx = rf_idx(idx0, sz)
	idx = mod(idx0, sz) + 1;
end