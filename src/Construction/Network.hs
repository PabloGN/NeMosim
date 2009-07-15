{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Construction.Network (
        Network(..),
        -- * Query
        size,
        synapseCount,
        indices,
        idxBounds,
        synapses,
        neurons,
        maxDelay,
        -- * Modify
        withNeurons,
        withTerminals,
        -- * Pretty-printing
        printConnections,
        printNeurons
    ) where

import Control.Monad (liftM2)
import Control.Parallel.Strategies (NFData, rnf)
import Data.Binary
import qualified Data.Map as Map
import Data.Maybe (fromJust)

import qualified Construction.Neuron as Neuron
import qualified Construction.Neurons as Neurons
import Construction.Synapse
import Construction.Topology
import Types



{- For the synapses we just store the indices of pre and post. The list should
 - be sorted to simplify the construction of the in-memory data later. -}
data Network n s = Network {
        networkNeurons     :: Neurons.Neurons n s,
        topology    :: Topology Idx
    } deriving (Eq, Show)


-------------------------------------------------------------------------------
-- Query
-------------------------------------------------------------------------------

{- | Return number of neurons in the network -}
size :: Network n s -> Int
size = Neurons.size . networkNeurons


{- | Return total number of synapses in the network -}
synapseCount :: Network n s -> Int
synapseCount = Neurons.synapseCount . networkNeurons


{- | Return indices of all valid neurons -}
indices :: Network n s -> [Idx]
indices = Neurons.indices . networkNeurons


{- | Return minimum and maximum neuron indices -}
idxBounds :: Network n s -> (Idx, Idx)
idxBounds = Neurons.idxBounds . networkNeurons


{- | Return synapses orderd by source and delay -}
synapses :: Network n s -> [(Idx, [(Delay, [(Idx, s)])])]
synapses = Neurons.synapses . networkNeurons


{- | Return list of all neurons -}
neurons :: Network n s -> [Neuron.Neuron n s]
-- TODO: merge Neurons into Network, this is just messy!
neurons = Neurons.neurons . networkNeurons


{- | Return maximum delay in network -}
maxDelay :: Network n s -> Delay
maxDelay = Neurons.maxDelay . networkNeurons



-------------------------------------------------------------------------------
-- Modification
-------------------------------------------------------------------------------


{- | Apply function to all neurons -}
-- TODO: perhaps use Neuron -> Neuron instead
withNeurons :: (Neurons.Neurons n s -> Neurons.Neurons n s) -> Network n s -> Network n s
withNeurons f (Network ns t) = (Network (f ns) t)


{- | Map function over all terminals (source and target) of all synapses -}
withTerminals :: (Idx -> Idx) -> Network n s -> Network n s
withTerminals f (Network ns t) = Network ns' t'
    where
        ns' = Neurons.withTerminals f ns
        t'  = fmap f t



-------------------------------------------------------------------------------
-- Various
-------------------------------------------------------------------------------

instance (Binary n, Binary s) => Binary (Network n s) where
    put (Network ns t) = put ns >> put t
    get = liftM2 Network get get


instance (NFData n, NFData s) => NFData (Network n s) where
    rnf (Network n t) = rnf n `seq` rnf t


-------------------------------------------------------------------------------
-- Printing
-------------------------------------------------------------------------------

printConnections :: (Show s) => Network n s -> IO ()
printConnections = Neurons.printConnections . networkNeurons

printNeurons :: (Show n, Show s) => Network n s -> IO ()
printNeurons = Neurons.printNeurons . networkNeurons
