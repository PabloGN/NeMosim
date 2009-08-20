{- State of simulation running on a CUDA device. Most of this is handled on the
 - c/cuda side of things, accessed throught Simulation.CUDA.KernelFFI -}

module Simulation.CUDA.State (State(..), CuRT) where

import Foreign.C.Types (CInt)
import Foreign.ForeignPtr (ForeignPtr)
import Simulation.CUDA.Address (ATT)

import Types (Delay)


{- Runtime data is managed on the CUDA-side in a single structure -}
data CuRT = CuRT


data State = State {
        pcount   :: Int,
        psize    :: [Int],          -- ^ size of each partition,
        maxDelay :: Delay,
        dt       :: CInt,           -- ^ number of steps in neuron update
        att      :: ATT,
        rt       :: ForeignPtr CuRT -- ^ kernel runtime data
    }