nemoPipelineLength: query the length of the simulation pipeline
---------------------------------------------------------------

::

	(INPUT, OUTPUT) = nemoPipelineLength

When executing in a pipelined fashion (see `nemoStepPL`) there are delays on both the input and output side of the simulation. INPUT is the number of simulation cycles that elapse between the user provides input to `nemoStepPL` and that input is used in the simulation. Similarily, OUTPUT is the number of simulation cycles that elapse between the simulation produces output and that output is made available to the user. The total delay is INPUT + OUTPUT. In a non-pipelined simulation both of these are zero. 
