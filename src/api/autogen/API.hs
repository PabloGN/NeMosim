module API where

import qualified Data.Map as M
import Data.Maybe

data Language = Matlab | MEX | CPP | C | LaTeX | Python deriving (Eq, Ord)

type ApiDescr = Maybe String

class Described a where
    describe :: a -> ApiDescr


type Name = String

class Named a where
    name :: a -> Name

instance Named Language where
    name Matlab = "Matlab"
    name MEX = "MEX"
    name CPP = "C++"
    name C = "C"

-- add c types here
data BaseType
        = ApiFloat
        | ApiUInt
        | ApiUInt64
        | ApiInt
        | ApiULong
        | ApiBool
        | ApiString
    deriving Eq

class Typed a where
    baseType :: a -> BaseType
    fullType :: a -> Type

-- instance Typed BaseType where
--    baseType = id


{- In some interfaces (esp. C) vector arguments need an extra associated length
 - argument. By default the length and vector/array are used as a pair.
 - Sometimes, however, the same length applies to several vectors, i.e. the
 - length is *implicit*. This should be documented in the associated
 - docstrings. For vectors with implicit length the s tring specifies the name
 - of the variable which length should be used (and checked against).
 -
 - The implicit dependency is always on some input argument.  -}
-- data VectorLength = ImplicitLength String | ExplicitLength deriving (Eq)
data VectorLength = ImplicitLength Input | ExplicitLength deriving (Eq)


explicitLength :: Type -> Bool
explicitLength (Scalar _) = False
explicitLength (Vector _ (ImplicitLength _)) = False
explicitLength (Vector _ ExplicitLength) = True


implicitLength :: Type -> Bool
implicitLength (Scalar _) = False
implicitLength (Vector _ (ImplicitLength _)) = True
implicitLength (Vector _ ExplicitLength) = False


data Type
        = Scalar BaseType
        | Vector BaseType VectorLength
    deriving Eq

instance Typed Type where
    baseType (Scalar t) = t
    baseType (Vector t _) = t
    fullType = id




instance Typed Input where
    baseType = baseType . arg
    fullType = fullType . arg




class Dimensional a where
    scalar :: a -> Bool
    vector :: a -> Bool
    scalar = not . vector

instance Dimensional Type where
    vector (Vector _ _) = True
    vector (Scalar _) = False



data ApiArg = ApiArg String ApiDescr Type deriving Eq
    -- = Scalar String ApiDescr BaseType
    -- | Vector String ApiDescr BaseType
    -- arguments can be grouped into vectors
    -- | VectorGroup [(String, Descr, Type)]

instance Named ApiArg where
    name (ApiArg n _ _) = n
    -- name (Scalar n _ _) = n
    -- name (Vector n _ _) = n

instance Dimensional ApiArg where
    vector = vector . arg_type

instance Typed ApiArg where
    baseType = baseType . arg_type
    fullType = arg_type

instance Described ApiArg where
    describe (ApiArg _ d _) = d

-- arg_descr (Scalar _ d _) = d
-- arg_descr (Vector _ d _) = d

arg_type :: ApiArg -> Type
arg_type (ApiArg _ _ t) = t


type OutputType = ApiArg

-- The optional argument default is just the string to insert into code
-- Need to add a value type otherwise
data Input
        = Required ApiArg
        | Optional ApiArg String
    deriving Eq

instance Dimensional Input where
    vector = vector . arg

instance Described Input where
    describe = describe . arg

arg :: Input -> ApiArg
arg (Required x) = x
arg (Optional x _) = x



instance Named Input where
    name = name . arg


{- Various aspsects of code generation differ on a per-language basis -}

{-
-- Some functions are written by hand
data Generation = Automatic | NoAutomatic

-- Some functions should come in vector form
data Vectorization = AutoVectorize | NoVectorization

data LanguageSpec = LanguageSpec {
        generation :: Generation,
        vectorization :: Vectorization
    }

defaultLanguageSpec = LanguageSpec Automatic | NoVectorization
-}


data ApiFunction = ApiFunction {
        fn_name :: String,       -- ^ canonical name
        fn_brief :: String,      -- ^ brief description of function
        -- TODO: change to a diffeent ApiDescr format that support construction
        -- of per-language strings from the ground up.
        fn_descr :: ApiDescr,    -- ^ detailed description of function
        fn_descr_extra :: M.Map Language String, -- ^ extra language-specific documentation
        fn_output :: [OutputType],
        fn_inputs :: [Input],
        fn_noauto :: [Language], -- ^ languages for which this function is hand-written
        fn_vectorized :: Bool    -- ^ vectorize this function where appropriate
    }


describeLanguage :: Language -> ApiFunction -> Maybe String
describeLanguage lang fn = if null combined then Nothing else Just combined
    where
        combined = concat $ catMaybes [main, extra]
        main = fn_descr fn
        extra = M.lookup lang $ fn_descr_extra fn


{- TODO: We really should have a more complex set of types above, to support
 - other constructor arguments -}
data Constructor
    = Constructor [ApiModule]
    | Factory [ApiModule]


defaultConstructor = Constructor []


{- The distincition between factory and regular constructor is not always important -}
constructorArgs :: Constructor -> [ApiModule]
constructorArgs (Constructor a) = a
constructorArgs (Factory a) = a





instance Named ApiFunction where
    name (ApiFunction n _ _ _ _ _ _ _) = n


instance Described ApiFunction where
    describe (ApiFunction _ _ d _ _ _ _ _) = d


inputCount :: ApiFunction -> Int
inputCount = length . fn_inputs


outputCount :: ApiFunction -> Int
outputCount = length . fn_output


data ApiModule = ApiModule {
        mdl_name :: String,             -- full name of module
        mdl_sname :: String,            -- short name used for variables
        mdl_descr :: ApiDescr,
        mdl_ctor :: Constructor,
        mdl_functions :: [ApiFunction],
        mdl_shared_functions :: [ApiFunction] -- ^ functions which are common to multiple modules
    }



instance Named ApiModule where
    name = mdl_name


instance Described ApiModule where
    describe = mdl_descr


allModuleFunctions :: ApiModule -> [ApiFunction]
allModuleFunctions mdl = (mdl_functions mdl) ++ (mdl_shared_functions mdl)



clearNetwork =
    ApiFunction
        "clearNetwork"
        "clear all neurons/synapses from network"
        Nothing
        M.empty
        []
        []
        [MEX]
        False



addNeuron =
    ApiFunction
        "addNeuron"
        "add a single neuron to the network"
        (Just "The neuron uses the Izhikevich neuron model. See E. M. Izhikevich \"Simple model of spiking neurons\", IEEE Trans. Neural Networks, vol 14, pp 1569-1572, 2003 for a full description of the model and the parameters.")
        M.empty
        []
        [   Required (ApiArg "idx" (Just "Neuron index (0-based)") (Scalar ApiUInt)),
            Required (ApiArg "a" (Just "Time scale of the recovery variable") (Scalar ApiFloat)),
            Required (ApiArg "b" (Just "Sensitivity to sub-threshold fluctuations in the membrane potential v") (Scalar ApiFloat)),
            Required (ApiArg "c" (Just "After-spike value of the membrane potential v") (Scalar ApiFloat)),
            Required (ApiArg "d" (Just "After-spike reset of the recovery variable u") (Scalar ApiFloat)),
            Required (ApiArg "u" (Just "Initial value for the membrane recovery variable") (Scalar ApiFloat)),
            Required (ApiArg "v" (Just "Initial value for the membrane potential") (Scalar ApiFloat)),
            Required (ApiArg "sigma" (Just "Parameter for a random gaussian per-neuron process which generates random input current drawn from an N(0, sigma) distribution. If set to zero no random input current will be generated") (Scalar ApiFloat))
        ]
        [] True


pythonVectorizedFull = [(Python, "The input arguments can be any combination of lists \
\ of equal length and scalars. If the input arguments contain a mix of scalars and lists \
\ the scalar arguments are replicated the required number of times.")]

pythonVectorized13 = [(Python, "The neuron and value parameters can be either both scalar or both lists of the same length")]

pythonVectorized1 = [(Python, "The neuron index may be either scalar or a list. The output has the same length as the neuron input")]


addSynapse =
    ApiFunction
        "addSynapse"
        "add a single synapse to the network"
        Nothing
        (M.fromList pythonVectorizedFull)
        [   ApiArg "id" (Just "Unique synapse ID") (Scalar ApiUInt64)]
        [   Required (ApiArg "source" (Just "Index of source neuron") (Scalar ApiUInt)),
            Required (ApiArg "target" (Just "Index of target neuron") (Scalar ApiUInt)),
            Required (ApiArg "delay" (Just "Synapse conductance delay in milliseconds") (Scalar ApiUInt)),
            Required (ApiArg "weight" (Just "Synapse weights") (Scalar ApiFloat)),
            Required (ApiArg "plastic" (Just "Boolean specifying whether or not this synapse is plastic") (Scalar ApiBool))
        ]
        [] True


getNeuronState =
    ApiFunction "getNeuronState"
        "get neuron state variable"
        (Just "For the Izhikevich model: 0=u, 1=v. ")
        (M.fromList pythonVectorized1)
        [   ApiArg "val" (Just "value of the relevant variable") (Scalar ApiFloat) ]
        [   Required (ApiArg "idx" (Just "neuron index") (Scalar ApiUInt)),
            Required (ApiArg "varno" (Just "variable index") (Scalar ApiUInt)) ]
        []
        False


getNeuronParameter =
    ApiFunction "getNeuronParameter"
        "get neuron parameter"
        (Just "The neuron parameters do not change during simulation. For the Izhikevich model: 0=a, 1=b, 2=c, 3=d. ")
        (M.fromList pythonVectorized1)
        [   ApiArg "val" (Just "value of the neuron parameter") (Scalar ApiFloat) ]
        [   Required (ApiArg "idx" (Just "neuron index") (Scalar ApiUInt)),
            Required (ApiArg "varno" (Just "variable index") (Scalar ApiUInt)) ]
        []
        False


setNeuronState =
    ApiFunction "setNeuronState"
        "set neuron state variable"
        (Just "For the Izhikevich model: 0=u, 1=v. ")
        (M.fromList pythonVectorized13)
        []
        [   Required (ApiArg "idx" (Just "neuron index") (Scalar ApiUInt)),
            Required (ApiArg "varno" (Just "variable index") (Scalar ApiUInt)),
            Required (ApiArg "val" (Just "value of the relevant variable") (Scalar ApiFloat)) ]
        []
        False


setNeuronParameter =
    ApiFunction "setNeuronParameter"
        "set neuron parameter"
        (Just "The neuron parameters do not change during simulation. For the Izhikevich model: 0=a, 1=b, 2=c, 3=d. ")
        (M.fromList pythonVectorized13)
        []
        [   Required (ApiArg "idx" (Just "neuron index") (Scalar ApiUInt)),
            Required (ApiArg "varno" (Just "variable index") (Scalar ApiUInt)),
            Required (ApiArg "val" (Just "value of the neuron parameter") (Scalar ApiFloat)) ]
        []
        False

neuronCount =
    ApiFunction
        "neuronCount"
        ""
        Nothing M.empty
        [ApiArg "ncount" (Just "number of neurons in the network") (Scalar ApiUInt)] [] [] False


network =
    ApiModule "Network" "net"
        (Just "A Network is constructed by adding individual neurons and synapses to the network. Neurons are given indices (from 0) which should be unique for each neuron. When adding synapses the source or target neurons need not necessarily exist yet, but should be defined before the network is finalised.")
        defaultConstructor
        [addNeuron, addSynapse, neuronCount, clearNetwork]
        constructable


step =
    let istim = Required $ ApiArg "istim_current"
                    (Just "The corresponding list of current input")
                    (Vector ApiFloat ExplicitLength)
    in ApiFunction "step"
        "run simulation for a single cycle (1ms)"
        Nothing M.empty
        [   ApiArg "fired" (Just "Neurons which fired this cycle") (Vector ApiUInt ExplicitLength) ]
        [   Required (ApiArg "fstim"
                (Just "An optional list of neurons, which will be forced to fire this cycle")
                (Vector ApiUInt ExplicitLength)),
            Required (ApiArg "istim_nidx"
                (Just "An optional list of neurons which will be given input current stimulus this cycle")
                (Vector ApiUInt (ImplicitLength istim))),
            istim
        ]
        [Matlab] False


applyStdp =
    ApiFunction "applyStdp"
        "update synapse weights using the accumulated STDP statistics"
        Nothing M.empty
        []
        [   Required (ApiArg "reward"
                (Just "Multiplier for the accumulated weight change")
                (Scalar ApiFloat)) ]
        [] False



setNeuron =
    ApiFunction "setNeuron"
        "modify an existing neuron"
        Nothing M.empty
        []
        [   Required (ApiArg "idx" (Just "Neuron index (0-based)") (Scalar ApiUInt)),
            Required (ApiArg "a" (Just "Time scale of the recovery variable") (Scalar ApiFloat)),
            Required (ApiArg "b" (Just "Sensitivity to sub-threshold fluctuations in the membrane potential v") (Scalar ApiFloat)),
            Required (ApiArg "c" (Just "After-spike value of the membrane potential v") (Scalar ApiFloat)),
            Required (ApiArg "d" (Just "After-spike reset of the recovery variable u") (Scalar ApiFloat)),
            Required (ApiArg "u" (Just "Initial value for the membrane recovery variable") (Scalar ApiFloat)),
            Required (ApiArg "v" (Just "Initial value for the membrane potential") (Scalar ApiFloat)),
            Required (ApiArg "sigma" (Just "Parameter for a random gaussian per-neuron process which generates random input current drawn from an N(0, sigma) distribution. If set to zero no random input current will be generated") (Scalar ApiFloat))
        ]
        [] True



getMembranePotential =
    ApiFunction "getMembranePotential"
        "get membane potential of a neuron"
        Nothing M.empty
        [   ApiArg "v" (Just "membrane potential") (Scalar ApiFloat) ]
        [   Required (ApiArg "idx" (Just "neuron index") (Scalar ApiUInt)) ]
        []
        True



getSynapsesFrom =
    ApiFunction "getSynapsesFrom"
        "return the synapse ids for all synapses with the given source neuron"
        Nothing M.empty
        [   ApiArg "synapses" (Just "synapse ids") (Vector ApiUInt64 ExplicitLength)]
        [   Required (ApiArg "source" (Just "source neuron index") (Scalar ApiUInt))]
        []
        False -- TODO: vectorize?


getTargets =
    let synapses = Required $ ApiArg "synapses"
                    (Just "synapse ids (as returned by addSynapse)")
                    (Vector ApiUInt64 ExplicitLength)
    in ApiFunction "getTargets"
        "return the targets for the specified synapses"
        -- TODO: add notes for C and C++ API, mentioning lifetime of returned pointer/reference
        Nothing M.empty
        [   ApiArg "targets" (Just "indices of target neurons") (Vector ApiUInt (ImplicitLength synapses)) ]
        [ synapses ]
        [] False




getDelays =
    let synapses = Required $ ApiArg "synapses"
                    (Just "synapse ids (as returned by addSynapse)")
                    (Vector ApiUInt64 ExplicitLength)
    in ApiFunction "getDelays"
        "return the conductance delays for the specified synapses"
        -- TODO: add notes for C and C++ API, mentioning lifetime of returned pointer/reference
        Nothing M.empty
        [   ApiArg "delays" (Just "conductance delays of the specified synpases") (Vector ApiUInt (ImplicitLength synapses)) ]
        [ synapses ]
        [] False


getWeights =
    let synapses = Required $ ApiArg "synapses"
                    (Just "synapse ids (as returned by addSynapse)")
                    (Vector ApiUInt64 ExplicitLength)
    in ApiFunction "getWeights"
        "return the weights for the specified synapses"
        -- TODO: add notes for C and C++ API, mentioning lifetime of returned pointer/reference
        Nothing M.empty
        [   ApiArg "weights" (Just "weights of the specified synapses") (Vector ApiFloat (ImplicitLength synapses)) ]
        [ synapses ]
        [] False



getPlastic =
    let synapses = Required $ ApiArg "synapses"
                    (Just "synapse ids (as returned by addSynapse)")
                    (Vector ApiUInt64 ExplicitLength)
    in ApiFunction "getPlastic"
        "return the boolean plasticity status for the specified synapses"
        -- TODO: add notes for C and C++ API, mentioning lifetime of returned pointer/reference
        Nothing M.empty
        [   ApiArg "plastic" (Just "plasticity status of the specified synpases") (Vector ApiBool (ImplicitLength synapses)) ]
        [ synapses ]
        [] False


elapsedWallclock =
    ApiFunction "elapsedWallclock" []
        Nothing M.empty
        [ApiArg "elapsed" (Just "number of milliseconds of wall-clock time elapsed since first simulation step (or last timer reset)") (Scalar ApiULong)]
        [] [] False


elapsedSimulation =
    ApiFunction "elapsedSimulation" []
        Nothing M.empty
        [ApiArg "elapsed" (Just "number of milliseconds of simulation time elapsed since first simulation step (or last timer reset)") (Scalar ApiULong)] [] [] False


resetTimer = ApiFunction "resetTimer" "reset both wall-clock and simulation timer" Nothing M.empty [] [] [] False

createSimulation =
    ApiFunction
        "createSimulation"
        "Initialise simulation data"
        (Just "Initialise simulation data, but do not start running. Call step to run simulation. The initialisation step can be time-consuming.")
        M.empty
        []
        []
        [MEX]
        False


destroySimulation =
    ApiFunction
        "destroySimulation"
        "Stop simulation and free associated data"
        (Just "The simulation can have a significant amount of memory associated with it. Calling destroySimulation frees up this memory.")
        M.empty
        []
        []
        [MEX]
        False


simulation =
    ApiModule "Simulation" "sim"
        (Just "A simulation is created from a network and a configuration object. The simulation is run by stepping through it, providing stimulus as appropriate. It is possible to read back synapse data at run time. The simulation also maintains a timer for both simulated time and wallclock time.")
        (Factory [network, configuration])
        [step, applyStdp, getMembranePotential,
            getSynapsesFrom, getTargets, getDelays, getWeights, getPlastic, elapsedWallclock, elapsedSimulation, resetTimer, createSimulation, destroySimulation]
        constructable



setCpuBackend =
    ApiFunction "setCpuBackend"
        "specify that the CPU backend should be used"
        (Just
            "Specify that the CPU backend should be used and optionally specify \
            \the number of threads to use. If the default thread count of -1 is \
            \used, the backend will choose a sensible value based on the available \
            \hardware concurrency."
        )
        M.empty
        []
        [   Optional (ApiArg "tcount" (Just "number of threads") (Scalar ApiInt)) "-1" ]
        [] False


setCudaBackend =
    ApiFunction "setCudaBackend"
        "specify that the CUDA backend should be used"
        (Just "Specify that the CUDA backend should be used and optionally specify \
\a desired device. If the (default) device value of -1 is used the \
\backend will choose the best available device. \
\ \
\ If the cuda backend (and the chosen device) cannot be used for \
\ whatever reason, an exception is raised. \
\ \
\ The device numbering is the numbering used internally by nemo (see \
\ cudaDeviceCount and cudaDeviceDescription). This device \
\ numbering may differ from the one provided by the CUDA driver \
\ directly, since NeMo ignores any devices it cannot use. "
        )
        M.empty
        []
        [   Optional (ApiArg "deviceNumber" Nothing (Scalar ApiInt)) "-1" ]
        [] False



setStdpFunction =
    ApiFunction "setStdpFunction" "enable STDP and set the global STDP function"
        -- TODO: add documentation here
        (Just "The STDP function is specified by providing the values sampled at integer cycles within the STDP window.")
        M.empty
        -- TODO: document limitations here
        []
        [   Required (ApiArg "prefire" (Just "STDP function values for spikes arrival times before the postsynaptic firing, starting closest to the postsynaptic firing") (Vector ApiFloat ExplicitLength)),
            Required (ApiArg "postfire" (Just "STDP function values for spikes arrival times after the postsynaptic firing, starting closest to the postsynaptic firing") (Vector ApiFloat ExplicitLength)),
            Required (ApiArg "minWeight" (Just "Lowest (negative) weight beyond which inhibitory synapses are not potentiated") (Scalar ApiFloat)),
            Required (ApiArg "maxWeight" (Just "Highest (positive) weight beyond which excitatory synapses are not potentiated") (Scalar ApiFloat))
        ] [] False


backendDescription =
    ApiFunction
        "backendDescription"
        "Description of the currently selected simulation backend"
        (Just "The backend can be changed using setCudaBackend or setCpuBackend")
        M.empty
        [ApiArg "description" (Just "Textual description of the currently selected backend") (Scalar ApiString)]
        []
        [] False


setWriteOnlySynapses =
    ApiFunction
        "setWriteOnlySynapses"
        "Specify that synapses will not be read back at run-time"
        (Just "By default synapse state can be read back at run-time. This may require setting up data structures of considerable size before starting the simulation. If the synapse state is not required at run-time, specify that synapses are write-only in order to save memory and setup time. By default synapses are readable")
        M.empty
        [] [] [] False


resetConfiguration =
    ApiFunction
        "resetConfiguration"
        "Replace configuration with default configuration"
        Nothing M.empty
        []
        []
        [MEX]
        False

configuration = ApiModule "Configuration" "conf" (Just "Global configuration") defaultConstructor
    [setCpuBackend, setCudaBackend, setStdpFunction, backendDescription, setWriteOnlySynapses, resetConfiguration]
    []



reset = ApiFunction "reset"
        "Reset all NeMo state, leaving an empty network, a default configuration, and no simulation"
        Nothing M.empty [] [] [MEX] False


matlabExtras = ApiModule "Others" "others" Nothing defaultConstructor [reset] []

{- | Some methods are common to the network and simulation modules. We deal with these methods separately -}
constructable :: [ApiFunction]
constructable = [setNeuron, setNeuronState, setNeuronParameter, getNeuronState, getNeuronParameter]