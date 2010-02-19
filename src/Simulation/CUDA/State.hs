{- State of simulation running on a CUDA device. Most of this is handled on the
 - c/cuda side of things, accessed throught Simulation.CUDA.KernelFFI -}

module Simulation.CUDA.State (State(..), CuRT) where

import Foreign.C.Types (CInt)
import Foreign.Ptr (Ptr)


{- Runtime data is managed on the CUDA-side in a single structure -}
data CuRT = CuRT


-- TODO: delete this whole file
data State = State {
        rt       :: Ptr CuRT -- ^ kernel runtime data
    }
