function nemoSetNeuron(idx, a, b, c, d, u, v, sigma)
% nemoSetNeuron - modify an existing neuron
%  
% Synopsis:
%   nemoSetNeuron(idx, a, b, c, d, u, v, sigma)
%  
% Inputs:
%   idx     - Neuron index (0-based)
%   a       - Time scale of the recovery variable
%   b       - Sensitivity to sub-threshold fluctuations in the membrane
%             potential v
%   c       - After-spike value of the membrane potential v
%   d       - After-spike reset of the recovery variable u
%   u       - Initial value for the membrane recovery variable
%   v       - Initial value for the membrane potential
%   sigma   - Parameter for a random gaussian per-neuron process which
%             generates random input current drawn from an N(0, sigma)
%             distribution. If set to zero no random input current will be
%             generated
%    
%  
% The inputs can be either all scalars or all vectors of the same
% length.
    nemo_mex(...
            uint32(24),...
            uint32(idx),...
            double(a),...
            double(b),...
            double(c),...
            double(d),...
            double(u),...
            double(v),...
            double(sigma)...
    );
end