{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}

{- | Common interface for simulator backends controlled via the FFI -}

module Simulation.CommonFFI (
    ForeignKernel(..),
    configureStdp,
    applyStdp
    )
where

import Control.Monad (when)
import Foreign.C.Types (CUInt)
import Foreign.Marshal.Array (withArray)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable)

import Simulation.STDP (StdpConf(..), prefireWindow, postfireWindow)


class (Fractional f, Storable f) => ForeignKernel rt f | rt -> f where

    ffi_enable_stdp
        :: Ptr rt -- ^ pointer to foreign data structure containing simulation runtime
        -> CUInt  -- ^ length of pre-fire part of STDP window
        -> CUInt  -- ^ length of post-fire part of STDP window
        -> Ptr f  -- ^ lookup-table values (dt -> float) for STDP function prefire,
        -> Ptr f  -- ^ lookup-table values (dt -> float) for STDP function postfire,
        -> f      -- ^ max weight: limit for excitatory synapses
        -> f      -- ^ min weight: limit for inhibitory synapses
        -> IO ()

    ffi_apply_stdp
        :: Ptr rt
        -> f      -- ^ reward
        -> IO ()



configureStdp :: ForeignKernel rt f => Ptr rt -> StdpConf -> IO ()
configureStdp rt conf =
    when (stdpEnabled conf) $ do
    withArray (map realToFrac $ prefire conf) $ \prefire_ptr -> do
    withArray (map realToFrac $ postfire conf) $ \postfire_ptr -> do
    ffi_enable_stdp rt
        (fromIntegral $ prefireWindow conf)
        (fromIntegral $ postfireWindow conf)
        prefire_ptr
        postfire_ptr
        (realToFrac $ stdpMaxWeight conf)
        (realToFrac $ stdpMinWeight conf)


applyStdp :: (ForeignKernel rt f, Fractional f) => Ptr rt -> Double -> IO ()
applyStdp rt reward = ffi_apply_stdp rt $ realToFrac reward