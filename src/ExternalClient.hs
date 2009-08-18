{-# LANGUAGE TypeSynonymInstances #-}

{- | Run nemo through an external thrift-based client interface -}

module ExternalClient (runExternalClient) where

import Control.Concurrent.MVar
import Control.Exception
import Control.Monad (liftM4)
import Data.Maybe (fromJust)
import Data.Map (Map)
import Thrift
import Thrift.Server

-- imports from auto-generated modules. These are not hierarchical
import NemoFrontend
import NemoFrontend_Iface
import qualified Nemo_Types as Wire

-- TODO: just import Construction base module
import qualified Construction.Network as Network
import Construction.Neurons
import Construction.Neuron
import Construction.Synapse
import Construction.Izhikevich
import Construction.Topology
import Options
import qualified Simulation as Backend (Simulation, Simulation_Iface(..))
import Simulation.CUDA.Options
import Simulation.Options
import Simulation.Backend (initSim)
import Simulation.STDP
import Simulation.STDP.Options
import Types



type Net = Network.Network (IzhNeuron Double) Static


-- TODO: add user configuration options

data NemoState
    = Constructing Net Config
    | Simulating Net Config Backend.Simulation


data Config = Config {
        stdpConfig :: StdpConf,
        simConfig :: SimulationOptions
    }


defaultConfig = Config (defaults stdpOptions) (defaults $ simOptions AllBackends)


{- | Return the /static/ network -}
network :: NemoState -> Net
network (Constructing net _) = net
network (Simulating net _ _) = net


{- | Create the initials state for construction -}
initNemoState :: NemoState
initNemoState = Constructing Network.empty defaultConfig


type ClientState = MVar NemoState


{- Apply function as long as we're in constuction mode -}
constructWith :: MVar NemoState -> (Net -> Net) -> IO ()
constructWith m f = do
    modifyMVar_ m $ \st -> do
    case st of
        Constructing net conf -> return $! Constructing (f net) conf
        _ -> fail "trying to construct during simulation"



{- Modify configuration, regardles of mode -}
reconfigure :: MVar NemoState -> (Config -> Config) -> IO ()
reconfigure m f = do
    modifyMVar_ m $ \st -> do
    case st of
        Constructing net conf -> return $! Constructing net (f conf)
        Simulating net conf sim -> return $! Simulating net (f conf) sim



{- To start simulation, freeze the network -}
startSimulation :: MVar NemoState -> IO ()
startSimulation m = do
    modifyMVar_ m $ \st -> do
    case st of
        Simulating _ _ _ -> return st
        Constructing net conf -> do
            sim <- initSim net simOpts cudaOpts (stdpConfig conf)
            return $! Simulating net conf sim
    where
        -- TODO: get values from 1) command-line 2) external client
        -- TODO: remove hard-coding
        -- TODO: move configuration into Config data type
        simOpts = SimulationOptions Forever 4 $ RemoteHost "localhost" 56100

        -- TODO: make this backend options instead of cudaOpts
        -- TODO: perhaps we want to be able to send this to server?
        cudaOpts = defaults cudaOptions



{- Apply simulation function. If nemo state is in construction mode, toggle this -} 
-- TODO: propagate errors to matlab
runSimulation :: [Wire.Stimulus] -> Backend.Simulation -> IO [Wire.Firing]
runSimulation stimulus sim = do
    pdata <- Backend.run sim $ map fstim stimulus
    return $! map firing pdata
    where
        fstim :: Wire.Stimulus -> [Idx]
        fstim = maybe (fail "invalid firing stimulus") id . Wire.f_Stimulus_firing

        firing :: ProbeData -> Wire.Firing
        firing (FiringData xs) = xs
        firing _ = error "runSimulation: non-firing data returned from simulation"



setStdpFn :: [Double] -> [Double] -> Double -> Config -> Config
setStdpFn prefire postfire maxWeight conf = conf { stdpConfig = stdpConfig' }
    where
        stdpConfig' = (stdpConfig conf) {
            stdpEnabled = True,
            prefire = prefire,
            postfire = postfire,
            stdpMaxWeight = maxWeight
        }


disableStdp' :: Config -> Config
disableStdp' conf = conf { stdpConfig = stdpConfig' }
    where
        stdpConfig' = (stdpConfig conf) { stdpEnabled = False }


setHost :: String -> Config -> Config
setHost host conf = conf { simConfig = simConfig' }
    where
        simConfig' = (simConfig conf) { optBackend = RemoteHost host 56100 }



getConnectivity' :: Backend.Simulation -> IO (Map Idx [Wire.Synapse])
getConnectivity' sim = do
    ss <- Backend.getWeights sim
    return $! fmap (map encodeSynapse) ss





simulateWith :: MVar NemoState -> (Backend.Simulation -> IO a) -> IO a
simulateWith m f = do
    modifyMVar m $ \st -> do
    case st of
        Simulating _ _ sim -> do
            ret <- f sim
            return $! (st, ret)
        Constructing net conf -> do
            -- Start simulation first
            putStrLn $ "starting simulation"
            sim <- initSim net (simConfig conf) cudaOpts (stdpConfig conf)
            ret <- f sim
            return $! (Simulating net conf sim, ret)
    where
        -- TODO: make this backend options instead of cudaOpts
        -- TODO: perhaps we want to be able to send this to server?
        cudaOpts = defaults cudaOptions




instance NemoFrontend_Iface ClientState where
    -- TODO: handle Maybes here!
    setBackend h (Just host) = reconfigure h $ setHost host
    setNetwork h (Just wnet) = constructWith h (\_ -> decodeNetwork wnet)
    run h (Just stim) = simulateWith h $ runSimulation stim
    enableStdp h (Just prefire) (Just postfire) (Just maxWeight) =
        reconfigure h $ setStdpFn prefire postfire maxWeight
    disableStdp h = reconfigure h $ disableStdp'
    applyStdp h (Just reward) = simulateWith h (\s -> Backend.applyStdp s reward)
    getConnectivity h = simulateWith h getConnectivity'


{- | Convert network from wire format to internal format -}
decodeNetwork :: Wire.IzhNetwork -> Net
decodeNetwork wnet = Network.Network ns t
    where
        ns = Neurons $ fmap decodeNeuron wnet
        t = Cluster $ map Node $ indices ns



{- | Convert neuron from wire format to internal format -}
decodeNeuron :: Wire.IzhNeuron -> Neuron (IzhNeuron Double) Static
decodeNeuron wn = neuron n ss
    where
        n = IzhNeuron
                (fromJust $ Wire.f_IzhNeuron_a wn)
                (fromJust $ Wire.f_IzhNeuron_b wn)
                (fromJust $ Wire.f_IzhNeuron_c wn)
                (fromJust $ Wire.f_IzhNeuron_d wn)
                (fromJust $ Wire.f_IzhNeuron_u wn)
                (fromJust $ Wire.f_IzhNeuron_v wn)
                0.0 False Nothing
        ss = map decodeSynapse $ fromJust $ Wire.f_IzhNeuron_axon wn


{- | Convert synapse from wire format to internal format -}
decodeSynapse :: Wire.Synapse -> Synapse Static
decodeSynapse ws = Synapse src tgt d $ Static w
    where
        src = fromJust $ Wire.f_Synapse_source ws
        tgt = fromJust $ Wire.f_Synapse_target ws
        d = fromJust $ Wire.f_Synapse_delay ws
        w = fromJust $ Wire.f_Synapse_weight ws


encodeSynapse :: Synapse Static -> Wire.Synapse
encodeSynapse s = Wire.Synapse src tgt d w
    where
        src = Just $ source s
        tgt = Just $ target s
        d   = Just $ delay s
        w   = Just $ current $ sdata s



runExternalClient :: IO ()
runExternalClient = do
    st <- newMVar initNemoState
    Control.Exception.handle (\(TransportExn s t) -> fail s) $ do
    runBasicServer st process 56101
