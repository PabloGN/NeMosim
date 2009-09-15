module Types where

import Data.List (sort)
import Control.Parallel.Strategies (rnf, NFData)

type FT = Double
-- type FT = Float

type Idx     = Int     -- unique indices for neurons
type Source  = Idx
type Target  = Idx
type Voltage = FT
type Current = FT
type Time    = Int     -- synchronous simulation only
type TemporalResolution = Int
type Delay   = Time

data Duration
        = Forever
        | Once
        | Until Time


{- Run-time probed data -}
-- TODO: rename: fire' -> fired (conflict with Neuron)
newtype FiringOutput = FiringOutput { fired' :: [Idx] } deriving (Eq)

instance Show FiringOutput where
    show (FiringOutput xs) = show xs


instance NFData FiringOutput where
    rnf (FiringOutput x) = rnf x `seq` ()
