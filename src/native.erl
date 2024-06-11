-module(native).
-export([ping/0, init_result/3, run_result/2, from_binary_result/3, to_binary_result/3, show_session_result/1, slice_result/4, reshape_result/2, concatenate_result/3]).
-nifs([init/3, run/2, from_binary/3, to_binary/3, show_session/1, slice/4, reshape/2, concatenate/3]).
-on_load(load/0).


load() ->
  io:fwrite("loading~n", []),
  ok = erlang:load_nif("native/ortex/target/debug/libortex", 0).

ping() ->
  io:fwrite("ping~n", []).
  % exit(nif_library_not_loaded).

nif_result_to_result({error, Val}) -> {error, Val};
nif_result_to_result(Val) -> {ok, Val}.
    

% When loading a NIF module, dummy clauses for all NIF function are required.
% NIF dummies usually just error out when called when the NIF is not loaded, as that should never normally happen.
init(_model_path, _execution_providers, _optimization_level) ->
  exit(nif_library_not_loaded).

init_result(model_path, execution_providers, optimization_level)->
  nif_result = init(model_path, execution_providers, optimization_level),
  nif_result_to_result(nif_result).

run(_model, _inputs) ->
  exit(nif_library_not_loaded).

run_result(model,inputs) ->
  nif_result = run(model,inputs),
  nif_result_to_result(nif_result).

from_binary(_bin, _shape, _type) ->
  exit(nif_library_not_loaded).

from_binary_result(_bin, _shape, _type) ->
  {ok , <<>>}.
  % nif_result = from_binary(bin, shape, type),
  % nif_result_to_result(nif_result).

to_binary(_reference, _bits, _limit) ->
  exit(nif_library_not_loaded).
to_binary_result(reference, bits, limit) ->
  nif_result = to_binary(reference, bits, limit),
  nif_result_to_result(nif_result).

show_session(_model) ->
  exit(nif_library_not_loaded).
show_session_result(model) ->
  nif_result = show_session(model),
  nif_result_to_result(nif_result).

slice(_tensor, _start_indicies, _lengths, _strides) ->
  exit(nif_library_not_loaded).
slice_result(tensor, start_indicies, lengths, strides) ->
  nif_result = slice(tensor, start_indicies, lengths, strides),
  nif_result_to_result(nif_result).

reshape(_tensor, _shape) ->
  exit(nif_library_not_loaded).
reshape_result(tensor, shape) ->
  nif_result = reshape(tensor, shape),
  nif_result_to_result(nif_result).


concatenate(_tensors_refs, _type, _axis) ->
  exit(nif_library_not_loaded).
concatenate_result(tensor_refs, type, axis) ->
  nif_result = concatenate(tensor_resfs, type, axis),
  nif_result_to_result(nif_result).
