-----------------------------------------------------------------
-- Autogenerated by Thrift                                     --
--                                                             --
-- DO NOT EDIT UNLESS YOU ARE SURE YOU KNOW WHAT YOU ARE DOING --
-----------------------------------------------------------------

module NemoBackend where
import Thrift
import Data.Typeable ( Typeable )
import Control.Exception
import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.Int
import Nemo_Types
import qualified NemoBackend_Iface as Iface
-- HELPER FUNCTIONS AND STRUCTURES --

data AddCluster_args = AddCluster_args{f_AddCluster_args_cluster :: Maybe [IzhNeuron]} deriving (Show,Eq,Ord,Typeable)
write_AddCluster_args oprot rec = do
  writeStructBegin oprot "AddCluster_args"
  case f_AddCluster_args_cluster rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("cluster",T_LIST,1)
    (let {f [] = return (); f (_viter212:t) = do {write_IzhNeuron oprot _viter212;f t}} in do {writeListBegin oprot (T_STRUCT,length _v); f _v;writeListEnd oprot})
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_AddCluster_args_fields iprot rec = do
  (_,_t214,_id215) <- readFieldBegin iprot
  if _t214 == T_STOP then return rec else
    case _id215 of 
      1 -> if _t214 == T_LIST then do
        s <- (let {f 0 = return []; f n = do {v <- (read_IzhNeuron iprot);r <- f (n-1); return $ v:r}} in do {(_etype219,_size216) <- readListBegin iprot; f _size216})
        read_AddCluster_args_fields iprot rec{f_AddCluster_args_cluster=Just s}
        else do
          skip iprot _t214
          read_AddCluster_args_fields iprot rec
      _ -> do
        skip iprot _t214
        readFieldEnd iprot
        read_AddCluster_args_fields iprot rec
read_AddCluster_args iprot = do
  readStructBegin iprot
  rec <- read_AddCluster_args_fields iprot (AddCluster_args{f_AddCluster_args_cluster=Nothing})
  readStructEnd iprot
  return rec
data AddCluster_result = AddCluster_result deriving (Show,Eq,Ord,Typeable)
write_AddCluster_result oprot rec = do
  writeStructBegin oprot "AddCluster_result"
  writeFieldStop oprot
  writeStructEnd oprot
read_AddCluster_result_fields iprot rec = do
  (_,_t224,_id225) <- readFieldBegin iprot
  if _t224 == T_STOP then return rec else
    case _id225 of 
      _ -> do
        skip iprot _t224
        readFieldEnd iprot
        read_AddCluster_result_fields iprot rec
read_AddCluster_result iprot = do
  readStructBegin iprot
  rec <- read_AddCluster_result_fields iprot (AddCluster_result{})
  readStructEnd iprot
  return rec
data AddNeuron_args = AddNeuron_args{f_AddNeuron_args_neuron :: Maybe IzhNeuron} deriving (Show,Eq,Ord,Typeable)
write_AddNeuron_args oprot rec = do
  writeStructBegin oprot "AddNeuron_args"
  case f_AddNeuron_args_neuron rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("neuron",T_STRUCT,1)
    write_IzhNeuron oprot _v
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_AddNeuron_args_fields iprot rec = do
  (_,_t229,_id230) <- readFieldBegin iprot
  if _t229 == T_STOP then return rec else
    case _id230 of 
      1 -> if _t229 == T_STRUCT then do
        s <- (read_IzhNeuron iprot)
        read_AddNeuron_args_fields iprot rec{f_AddNeuron_args_neuron=Just s}
        else do
          skip iprot _t229
          read_AddNeuron_args_fields iprot rec
      _ -> do
        skip iprot _t229
        readFieldEnd iprot
        read_AddNeuron_args_fields iprot rec
read_AddNeuron_args iprot = do
  readStructBegin iprot
  rec <- read_AddNeuron_args_fields iprot (AddNeuron_args{f_AddNeuron_args_neuron=Nothing})
  readStructEnd iprot
  return rec
data AddNeuron_result = AddNeuron_result{f_AddNeuron_result_err :: Maybe ConstructionError} deriving (Show,Eq,Ord,Typeable)
write_AddNeuron_result oprot rec = do
  writeStructBegin oprot "AddNeuron_result"
  case f_AddNeuron_result_err rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("err",T_STRUCT,1)
    write_ConstructionError oprot _v
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_AddNeuron_result_fields iprot rec = do
  (_,_t234,_id235) <- readFieldBegin iprot
  if _t234 == T_STOP then return rec else
    case _id235 of 
      1 -> if _t234 == T_STRUCT then do
        s <- (read_ConstructionError iprot)
        read_AddNeuron_result_fields iprot rec{f_AddNeuron_result_err=Just s}
        else do
          skip iprot _t234
          read_AddNeuron_result_fields iprot rec
      _ -> do
        skip iprot _t234
        readFieldEnd iprot
        read_AddNeuron_result_fields iprot rec
read_AddNeuron_result iprot = do
  readStructBegin iprot
  rec <- read_AddNeuron_result_fields iprot (AddNeuron_result{f_AddNeuron_result_err=Nothing})
  readStructEnd iprot
  return rec
data EnableStdp_args = EnableStdp_args{f_EnableStdp_args_prefire :: Maybe [Double],f_EnableStdp_args_postfire :: Maybe [Double],f_EnableStdp_args_maxWeight :: Maybe Double,f_EnableStdp_args_minWeight :: Maybe Double} deriving (Show,Eq,Ord,Typeable)
write_EnableStdp_args oprot rec = do
  writeStructBegin oprot "EnableStdp_args"
  case f_EnableStdp_args_prefire rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("prefire",T_LIST,1)
    (let {f [] = return (); f (_viter238:t) = do {writeDouble oprot _viter238;f t}} in do {writeListBegin oprot (T_DOUBLE,length _v); f _v;writeListEnd oprot})
    writeFieldEnd oprot}
  case f_EnableStdp_args_postfire rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("postfire",T_LIST,2)
    (let {f [] = return (); f (_viter239:t) = do {writeDouble oprot _viter239;f t}} in do {writeListBegin oprot (T_DOUBLE,length _v); f _v;writeListEnd oprot})
    writeFieldEnd oprot}
  case f_EnableStdp_args_maxWeight rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("maxWeight",T_DOUBLE,3)
    writeDouble oprot _v
    writeFieldEnd oprot}
  case f_EnableStdp_args_minWeight rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("minWeight",T_DOUBLE,4)
    writeDouble oprot _v
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_EnableStdp_args_fields iprot rec = do
  (_,_t241,_id242) <- readFieldBegin iprot
  if _t241 == T_STOP then return rec else
    case _id242 of 
      1 -> if _t241 == T_LIST then do
        s <- (let {f 0 = return []; f n = do {v <- readDouble iprot;r <- f (n-1); return $ v:r}} in do {(_etype246,_size243) <- readListBegin iprot; f _size243})
        read_EnableStdp_args_fields iprot rec{f_EnableStdp_args_prefire=Just s}
        else do
          skip iprot _t241
          read_EnableStdp_args_fields iprot rec
      2 -> if _t241 == T_LIST then do
        s <- (let {f 0 = return []; f n = do {v <- readDouble iprot;r <- f (n-1); return $ v:r}} in do {(_etype251,_size248) <- readListBegin iprot; f _size248})
        read_EnableStdp_args_fields iprot rec{f_EnableStdp_args_postfire=Just s}
        else do
          skip iprot _t241
          read_EnableStdp_args_fields iprot rec
      3 -> if _t241 == T_DOUBLE then do
        s <- readDouble iprot
        read_EnableStdp_args_fields iprot rec{f_EnableStdp_args_maxWeight=Just s}
        else do
          skip iprot _t241
          read_EnableStdp_args_fields iprot rec
      4 -> if _t241 == T_DOUBLE then do
        s <- readDouble iprot
        read_EnableStdp_args_fields iprot rec{f_EnableStdp_args_minWeight=Just s}
        else do
          skip iprot _t241
          read_EnableStdp_args_fields iprot rec
      _ -> do
        skip iprot _t241
        readFieldEnd iprot
        read_EnableStdp_args_fields iprot rec
read_EnableStdp_args iprot = do
  readStructBegin iprot
  rec <- read_EnableStdp_args_fields iprot (EnableStdp_args{f_EnableStdp_args_prefire=Nothing,f_EnableStdp_args_postfire=Nothing,f_EnableStdp_args_maxWeight=Nothing,f_EnableStdp_args_minWeight=Nothing})
  readStructEnd iprot
  return rec
data EnableStdp_result = EnableStdp_result deriving (Show,Eq,Ord,Typeable)
write_EnableStdp_result oprot rec = do
  writeStructBegin oprot "EnableStdp_result"
  writeFieldStop oprot
  writeStructEnd oprot
read_EnableStdp_result_fields iprot rec = do
  (_,_t256,_id257) <- readFieldBegin iprot
  if _t256 == T_STOP then return rec else
    case _id257 of 
      _ -> do
        skip iprot _t256
        readFieldEnd iprot
        read_EnableStdp_result_fields iprot rec
read_EnableStdp_result iprot = do
  readStructBegin iprot
  rec <- read_EnableStdp_result_fields iprot (EnableStdp_result{})
  readStructEnd iprot
  return rec
data EnablePipelining_args = EnablePipelining_args deriving (Show,Eq,Ord,Typeable)
write_EnablePipelining_args oprot rec = do
  writeStructBegin oprot "EnablePipelining_args"
  writeFieldStop oprot
  writeStructEnd oprot
read_EnablePipelining_args_fields iprot rec = do
  (_,_t261,_id262) <- readFieldBegin iprot
  if _t261 == T_STOP then return rec else
    case _id262 of 
      _ -> do
        skip iprot _t261
        readFieldEnd iprot
        read_EnablePipelining_args_fields iprot rec
read_EnablePipelining_args iprot = do
  readStructBegin iprot
  rec <- read_EnablePipelining_args_fields iprot (EnablePipelining_args{})
  readStructEnd iprot
  return rec
data EnablePipelining_result = EnablePipelining_result deriving (Show,Eq,Ord,Typeable)
write_EnablePipelining_result oprot rec = do
  writeStructBegin oprot "EnablePipelining_result"
  writeFieldStop oprot
  writeStructEnd oprot
read_EnablePipelining_result_fields iprot rec = do
  (_,_t266,_id267) <- readFieldBegin iprot
  if _t266 == T_STOP then return rec else
    case _id267 of 
      _ -> do
        skip iprot _t266
        readFieldEnd iprot
        read_EnablePipelining_result_fields iprot rec
read_EnablePipelining_result iprot = do
  readStructBegin iprot
  rec <- read_EnablePipelining_result_fields iprot (EnablePipelining_result{})
  readStructEnd iprot
  return rec
data PipelineLength_args = PipelineLength_args deriving (Show,Eq,Ord,Typeable)
write_PipelineLength_args oprot rec = do
  writeStructBegin oprot "PipelineLength_args"
  writeFieldStop oprot
  writeStructEnd oprot
read_PipelineLength_args_fields iprot rec = do
  (_,_t271,_id272) <- readFieldBegin iprot
  if _t271 == T_STOP then return rec else
    case _id272 of 
      _ -> do
        skip iprot _t271
        readFieldEnd iprot
        read_PipelineLength_args_fields iprot rec
read_PipelineLength_args iprot = do
  readStructBegin iprot
  rec <- read_PipelineLength_args_fields iprot (PipelineLength_args{})
  readStructEnd iprot
  return rec
data PipelineLength_result = PipelineLength_result{f_PipelineLength_result_success :: Maybe PipelineLength} deriving (Show,Eq,Ord,Typeable)
write_PipelineLength_result oprot rec = do
  writeStructBegin oprot "PipelineLength_result"
  case f_PipelineLength_result_success rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("success",T_STRUCT,0)
    write_PipelineLength oprot _v
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_PipelineLength_result_fields iprot rec = do
  (_,_t276,_id277) <- readFieldBegin iprot
  if _t276 == T_STOP then return rec else
    case _id277 of 
      0 -> if _t276 == T_STRUCT then do
        s <- (read_PipelineLength iprot)
        read_PipelineLength_result_fields iprot rec{f_PipelineLength_result_success=Just s}
        else do
          skip iprot _t276
          read_PipelineLength_result_fields iprot rec
      _ -> do
        skip iprot _t276
        readFieldEnd iprot
        read_PipelineLength_result_fields iprot rec
read_PipelineLength_result iprot = do
  readStructBegin iprot
  rec <- read_PipelineLength_result_fields iprot (PipelineLength_result{f_PipelineLength_result_success=Nothing})
  readStructEnd iprot
  return rec
data StartSimulation_args = StartSimulation_args deriving (Show,Eq,Ord,Typeable)
write_StartSimulation_args oprot rec = do
  writeStructBegin oprot "StartSimulation_args"
  writeFieldStop oprot
  writeStructEnd oprot
read_StartSimulation_args_fields iprot rec = do
  (_,_t281,_id282) <- readFieldBegin iprot
  if _t281 == T_STOP then return rec else
    case _id282 of 
      _ -> do
        skip iprot _t281
        readFieldEnd iprot
        read_StartSimulation_args_fields iprot rec
read_StartSimulation_args iprot = do
  readStructBegin iprot
  rec <- read_StartSimulation_args_fields iprot (StartSimulation_args{})
  readStructEnd iprot
  return rec
data StartSimulation_result = StartSimulation_result{f_StartSimulation_result_err :: Maybe ConstructionError} deriving (Show,Eq,Ord,Typeable)
write_StartSimulation_result oprot rec = do
  writeStructBegin oprot "StartSimulation_result"
  case f_StartSimulation_result_err rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("err",T_STRUCT,1)
    write_ConstructionError oprot _v
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_StartSimulation_result_fields iprot rec = do
  (_,_t286,_id287) <- readFieldBegin iprot
  if _t286 == T_STOP then return rec else
    case _id287 of 
      1 -> if _t286 == T_STRUCT then do
        s <- (read_ConstructionError iprot)
        read_StartSimulation_result_fields iprot rec{f_StartSimulation_result_err=Just s}
        else do
          skip iprot _t286
          read_StartSimulation_result_fields iprot rec
      _ -> do
        skip iprot _t286
        readFieldEnd iprot
        read_StartSimulation_result_fields iprot rec
read_StartSimulation_result iprot = do
  readStructBegin iprot
  rec <- read_StartSimulation_result_fields iprot (StartSimulation_result{f_StartSimulation_result_err=Nothing})
  readStructEnd iprot
  return rec
data Run_args = Run_args{f_Run_args_stim :: Maybe [Stimulus]} deriving (Show,Eq,Ord,Typeable)
write_Run_args oprot rec = do
  writeStructBegin oprot "Run_args"
  case f_Run_args_stim rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("stim",T_LIST,1)
    (let {f [] = return (); f (_viter290:t) = do {write_Stimulus oprot _viter290;f t}} in do {writeListBegin oprot (T_STRUCT,length _v); f _v;writeListEnd oprot})
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_Run_args_fields iprot rec = do
  (_,_t292,_id293) <- readFieldBegin iprot
  if _t292 == T_STOP then return rec else
    case _id293 of 
      1 -> if _t292 == T_LIST then do
        s <- (let {f 0 = return []; f n = do {v <- (read_Stimulus iprot);r <- f (n-1); return $ v:r}} in do {(_etype297,_size294) <- readListBegin iprot; f _size294})
        read_Run_args_fields iprot rec{f_Run_args_stim=Just s}
        else do
          skip iprot _t292
          read_Run_args_fields iprot rec
      _ -> do
        skip iprot _t292
        readFieldEnd iprot
        read_Run_args_fields iprot rec
read_Run_args iprot = do
  readStructBegin iprot
  rec <- read_Run_args_fields iprot (Run_args{f_Run_args_stim=Nothing})
  readStructEnd iprot
  return rec
data Run_result = Run_result{f_Run_result_success :: Maybe [[Int]],f_Run_result_err :: Maybe ConstructionError} deriving (Show,Eq,Ord,Typeable)
write_Run_result oprot rec = do
  writeStructBegin oprot "Run_result"
  case f_Run_result_success rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("success",T_LIST,0)
    (let {f [] = return (); f (_viter301:t) = do {(let {f [] = return (); f (_viter302:t) = do {writeI32 oprot _viter302;f t}} in do {writeListBegin oprot (T_I32,length _viter301); f _viter301;writeListEnd oprot});f t}} in do {writeListBegin oprot (T_LIST,length _v); f _v;writeListEnd oprot})
    writeFieldEnd oprot}
  case f_Run_result_err rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("err",T_STRUCT,1)
    write_ConstructionError oprot _v
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_Run_result_fields iprot rec = do
  (_,_t304,_id305) <- readFieldBegin iprot
  if _t304 == T_STOP then return rec else
    case _id305 of 
      0 -> if _t304 == T_LIST then do
        s <- (let {f 0 = return []; f n = do {v <- (let {f 0 = return []; f n = do {v <- readI32 iprot;r <- f (n-1); return $ v:r}} in do {(_etype314,_size311) <- readListBegin iprot; f _size311});r <- f (n-1); return $ v:r}} in do {(_etype309,_size306) <- readListBegin iprot; f _size306})
        read_Run_result_fields iprot rec{f_Run_result_success=Just s}
        else do
          skip iprot _t304
          read_Run_result_fields iprot rec
      1 -> if _t304 == T_STRUCT then do
        s <- (read_ConstructionError iprot)
        read_Run_result_fields iprot rec{f_Run_result_err=Just s}
        else do
          skip iprot _t304
          read_Run_result_fields iprot rec
      _ -> do
        skip iprot _t304
        readFieldEnd iprot
        read_Run_result_fields iprot rec
read_Run_result iprot = do
  readStructBegin iprot
  rec <- read_Run_result_fields iprot (Run_result{f_Run_result_success=Nothing,f_Run_result_err=Nothing})
  readStructEnd iprot
  return rec
data ApplyStdp_args = ApplyStdp_args{f_ApplyStdp_args_reward :: Maybe Double} deriving (Show,Eq,Ord,Typeable)
write_ApplyStdp_args oprot rec = do
  writeStructBegin oprot "ApplyStdp_args"
  case f_ApplyStdp_args_reward rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("reward",T_DOUBLE,1)
    writeDouble oprot _v
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_ApplyStdp_args_fields iprot rec = do
  (_,_t319,_id320) <- readFieldBegin iprot
  if _t319 == T_STOP then return rec else
    case _id320 of 
      1 -> if _t319 == T_DOUBLE then do
        s <- readDouble iprot
        read_ApplyStdp_args_fields iprot rec{f_ApplyStdp_args_reward=Just s}
        else do
          skip iprot _t319
          read_ApplyStdp_args_fields iprot rec
      _ -> do
        skip iprot _t319
        readFieldEnd iprot
        read_ApplyStdp_args_fields iprot rec
read_ApplyStdp_args iprot = do
  readStructBegin iprot
  rec <- read_ApplyStdp_args_fields iprot (ApplyStdp_args{f_ApplyStdp_args_reward=Nothing})
  readStructEnd iprot
  return rec
data ApplyStdp_result = ApplyStdp_result{f_ApplyStdp_result_err :: Maybe ConstructionError} deriving (Show,Eq,Ord,Typeable)
write_ApplyStdp_result oprot rec = do
  writeStructBegin oprot "ApplyStdp_result"
  case f_ApplyStdp_result_err rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("err",T_STRUCT,1)
    write_ConstructionError oprot _v
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_ApplyStdp_result_fields iprot rec = do
  (_,_t324,_id325) <- readFieldBegin iprot
  if _t324 == T_STOP then return rec else
    case _id325 of 
      1 -> if _t324 == T_STRUCT then do
        s <- (read_ConstructionError iprot)
        read_ApplyStdp_result_fields iprot rec{f_ApplyStdp_result_err=Just s}
        else do
          skip iprot _t324
          read_ApplyStdp_result_fields iprot rec
      _ -> do
        skip iprot _t324
        readFieldEnd iprot
        read_ApplyStdp_result_fields iprot rec
read_ApplyStdp_result iprot = do
  readStructBegin iprot
  rec <- read_ApplyStdp_result_fields iprot (ApplyStdp_result{f_ApplyStdp_result_err=Nothing})
  readStructEnd iprot
  return rec
data GetConnectivity_args = GetConnectivity_args deriving (Show,Eq,Ord,Typeable)
write_GetConnectivity_args oprot rec = do
  writeStructBegin oprot "GetConnectivity_args"
  writeFieldStop oprot
  writeStructEnd oprot
read_GetConnectivity_args_fields iprot rec = do
  (_,_t329,_id330) <- readFieldBegin iprot
  if _t329 == T_STOP then return rec else
    case _id330 of 
      _ -> do
        skip iprot _t329
        readFieldEnd iprot
        read_GetConnectivity_args_fields iprot rec
read_GetConnectivity_args iprot = do
  readStructBegin iprot
  rec <- read_GetConnectivity_args_fields iprot (GetConnectivity_args{})
  readStructEnd iprot
  return rec
data GetConnectivity_result = GetConnectivity_result{f_GetConnectivity_result_success :: Maybe (Map.Map Int [Synapse])} deriving (Show,Eq,Ord,Typeable)
write_GetConnectivity_result oprot rec = do
  writeStructBegin oprot "GetConnectivity_result"
  case f_GetConnectivity_result_success rec of {Nothing -> return (); Just _v -> do
    writeFieldBegin oprot ("success",T_MAP,0)
    (let {f [] = return (); f ((_kiter333,_viter334):t) = do {do {writeI32 oprot _kiter333;(let {f [] = return (); f (_viter335:t) = do {write_Synapse oprot _viter335;f t}} in do {writeListBegin oprot (T_STRUCT,length _viter334); f _viter334;writeListEnd oprot})};f t}} in do {writeMapBegin oprot (T_I32,T_LIST,Map.size _v); f (Map.toList _v);writeMapEnd oprot})
    writeFieldEnd oprot}
  writeFieldStop oprot
  writeStructEnd oprot
read_GetConnectivity_result_fields iprot rec = do
  (_,_t337,_id338) <- readFieldBegin iprot
  if _t337 == T_STOP then return rec else
    case _id338 of 
      0 -> if _t337 == T_MAP then do
        s <- (let {f 0 = return []; f n = do {k <- readI32 iprot; v <- (let {f 0 = return []; f n = do {v <- (read_Synapse iprot);r <- f (n-1); return $ v:r}} in do {(_etype347,_size344) <- readListBegin iprot; f _size344});r <- f (n-1); return $ (k,v):r}} in do {(_ktype340,_vtype341,_size339) <- readMapBegin iprot; l <- f _size339; return $ Map.fromList l})
        read_GetConnectivity_result_fields iprot rec{f_GetConnectivity_result_success=Just s}
        else do
          skip iprot _t337
          read_GetConnectivity_result_fields iprot rec
      _ -> do
        skip iprot _t337
        readFieldEnd iprot
        read_GetConnectivity_result_fields iprot rec
read_GetConnectivity_result iprot = do
  readStructBegin iprot
  rec <- read_GetConnectivity_result_fields iprot (GetConnectivity_result{f_GetConnectivity_result_success=Nothing})
  readStructEnd iprot
  return rec
data StopSimulation_args = StopSimulation_args deriving (Show,Eq,Ord,Typeable)
write_StopSimulation_args oprot rec = do
  writeStructBegin oprot "StopSimulation_args"
  writeFieldStop oprot
  writeStructEnd oprot
read_StopSimulation_args_fields iprot rec = do
  (_,_t352,_id353) <- readFieldBegin iprot
  if _t352 == T_STOP then return rec else
    case _id353 of 
      _ -> do
        skip iprot _t352
        readFieldEnd iprot
        read_StopSimulation_args_fields iprot rec
read_StopSimulation_args iprot = do
  readStructBegin iprot
  rec <- read_StopSimulation_args_fields iprot (StopSimulation_args{})
  readStructEnd iprot
  return rec
data StopSimulation_result = StopSimulation_result deriving (Show,Eq,Ord,Typeable)
write_StopSimulation_result oprot rec = do
  writeStructBegin oprot "StopSimulation_result"
  writeFieldStop oprot
  writeStructEnd oprot
read_StopSimulation_result_fields iprot rec = do
  (_,_t357,_id358) <- readFieldBegin iprot
  if _t357 == T_STOP then return rec else
    case _id358 of 
      _ -> do
        skip iprot _t357
        readFieldEnd iprot
        read_StopSimulation_result_fields iprot rec
read_StopSimulation_result iprot = do
  readStructBegin iprot
  rec <- read_StopSimulation_result_fields iprot (StopSimulation_result{})
  readStructEnd iprot
  return rec
process_addCluster (seqid, iprot, oprot, handler) = do
  args <- read_AddCluster_args iprot
  readMessageEnd iprot
  rs <- return (AddCluster_result)
  res <- (do
    Iface.addCluster handler (f_AddCluster_args_cluster args)
    return rs)
  writeMessageBegin oprot ("addCluster", M_REPLY, seqid);
  write_AddCluster_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_addNeuron (seqid, iprot, oprot, handler) = do
  args <- read_AddNeuron_args iprot
  readMessageEnd iprot
  rs <- return (AddNeuron_result Nothing)
  res <- (Control.Exception.catch
    (do
      Iface.addNeuron handler (f_AddNeuron_args_neuron args)
      return rs)
    (\e  -> 
      return rs{f_AddNeuron_result_err =Just e}))
  writeMessageBegin oprot ("addNeuron", M_REPLY, seqid);
  write_AddNeuron_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_enableStdp (seqid, iprot, oprot, handler) = do
  args <- read_EnableStdp_args iprot
  readMessageEnd iprot
  rs <- return (EnableStdp_result)
  res <- (do
    Iface.enableStdp handler (f_EnableStdp_args_prefire args) (f_EnableStdp_args_postfire args) (f_EnableStdp_args_maxWeight args) (f_EnableStdp_args_minWeight args)
    return rs)
  writeMessageBegin oprot ("enableStdp", M_REPLY, seqid);
  write_EnableStdp_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_enablePipelining (seqid, iprot, oprot, handler) = do
  args <- read_EnablePipelining_args iprot
  readMessageEnd iprot
  rs <- return (EnablePipelining_result)
  res <- (do
    Iface.enablePipelining handler
    return rs)
  writeMessageBegin oprot ("enablePipelining", M_REPLY, seqid);
  write_EnablePipelining_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_pipelineLength (seqid, iprot, oprot, handler) = do
  args <- read_PipelineLength_args iprot
  readMessageEnd iprot
  rs <- return (PipelineLength_result Nothing)
  res <- (do
    res <- Iface.pipelineLength handler
    return rs{f_PipelineLength_result_success= Just res})
  writeMessageBegin oprot ("pipelineLength", M_REPLY, seqid);
  write_PipelineLength_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_startSimulation (seqid, iprot, oprot, handler) = do
  args <- read_StartSimulation_args iprot
  readMessageEnd iprot
  rs <- return (StartSimulation_result Nothing)
  res <- (Control.Exception.catch
    (do
      Iface.startSimulation handler
      return rs)
    (\e  -> 
      return rs{f_StartSimulation_result_err =Just e}))
  writeMessageBegin oprot ("startSimulation", M_REPLY, seqid);
  write_StartSimulation_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_run (seqid, iprot, oprot, handler) = do
  args <- read_Run_args iprot
  readMessageEnd iprot
  rs <- return (Run_result Nothing Nothing)
  res <- (Control.Exception.catch
    (do
      res <- Iface.run handler (f_Run_args_stim args)
      return rs{f_Run_result_success= Just res})
    (\e  -> 
      return rs{f_Run_result_err =Just e}))
  writeMessageBegin oprot ("run", M_REPLY, seqid);
  write_Run_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_applyStdp (seqid, iprot, oprot, handler) = do
  args <- read_ApplyStdp_args iprot
  readMessageEnd iprot
  rs <- return (ApplyStdp_result Nothing)
  res <- (Control.Exception.catch
    (do
      Iface.applyStdp handler (f_ApplyStdp_args_reward args)
      return rs)
    (\e  -> 
      return rs{f_ApplyStdp_result_err =Just e}))
  writeMessageBegin oprot ("applyStdp", M_REPLY, seqid);
  write_ApplyStdp_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_getConnectivity (seqid, iprot, oprot, handler) = do
  args <- read_GetConnectivity_args iprot
  readMessageEnd iprot
  rs <- return (GetConnectivity_result Nothing)
  res <- (do
    res <- Iface.getConnectivity handler
    return rs{f_GetConnectivity_result_success= Just res})
  writeMessageBegin oprot ("getConnectivity", M_REPLY, seqid);
  write_GetConnectivity_result oprot res
  writeMessageEnd oprot
  tFlush (getTransport oprot)
process_stopSimulation (seqid, iprot, oprot, handler) = do
  args <- read_StopSimulation_args iprot
  readMessageEnd iprot
  res <- (do
    Iface.stopSimulation handler)
  return ()
proc handler (iprot,oprot) (name,typ,seqid) = case name of
  "addCluster" -> process_addCluster (seqid,iprot,oprot,handler)
  "addNeuron" -> process_addNeuron (seqid,iprot,oprot,handler)
  "enableStdp" -> process_enableStdp (seqid,iprot,oprot,handler)
  "enablePipelining" -> process_enablePipelining (seqid,iprot,oprot,handler)
  "pipelineLength" -> process_pipelineLength (seqid,iprot,oprot,handler)
  "startSimulation" -> process_startSimulation (seqid,iprot,oprot,handler)
  "run" -> process_run (seqid,iprot,oprot,handler)
  "applyStdp" -> process_applyStdp (seqid,iprot,oprot,handler)
  "getConnectivity" -> process_getConnectivity (seqid,iprot,oprot,handler)
  "stopSimulation" -> process_stopSimulation (seqid,iprot,oprot,handler)
  _ -> do
    skip iprot T_STRUCT
    readMessageEnd iprot
    writeMessageBegin oprot (name,M_EXCEPTION,seqid)
    writeAppExn oprot (AppExn AE_UNKNOWN_METHOD ("Unknown function " ++ name))
    writeMessageEnd oprot
    tFlush (getTransport oprot)
process handler (iprot, oprot) = do
  (name, typ, seqid) <- readMessageBegin iprot
  proc handler (iprot,oprot) (name,typ,seqid)
  return True