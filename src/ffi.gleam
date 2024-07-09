import gleam/dynamic.{type Dynamic}
import gleam/erlang/atom.{type Atom}
import gleam/io
import gleam/option.{type Option}

pub type OrtModel

pub type OrtTensor

pub type NifResult(val, err)

@external(erlang, "native", "nif_result_to_result")
pub fn nif_result_to_result(nif: NifResult(val, err)) -> Result(val, err)

@external(erlang, "native", "ping")
pub fn ping() -> Nil

@external(erlang, "native", "init")
pub fn init_nif(
  path: String,
  eps: List(Atom),
  opt: Int,
) -> NifResult(OrtModel, String)

pub fn init(path: String, eps: List(Atom), opt: Int) -> Result(OrtModel, String) {
  init_nif(path, eps, opt) |> nif_result_to_result
}

@external(erlang, "native", "run")
pub fn run_nif(
  model: OrtModel,
  inputs: List(OrtTensor),
) -> NifResult(List(#(OrtTensor, List(Int), atom.Atom, Int)), String)

pub fn run(
  model: OrtModel,
  inputs: List(OrtTensor),
) -> Result(List(#(OrtTensor, List(Int), atom.Atom, Int)), String) {
  run_nif(model, inputs) |> nif_result_to_result
}

@external(erlang, "native", "show_ession")
pub fn show_session_nif(
  model: OrtModel,
) -> Result(
  #(
    List(#(String, String, Option(List(Int)))),
    List(#(String, String, Option(List(Int)))),
  ),
  String,
)

@external(erlang, "native", "from_binary")
fn from_binary_nif(
  bin: BitArray,
  shape: List(Int),
  dtype: #(atom.Atom, Int),
) -> NifResult(OrtTensor, String)

pub fn from_binary(bin: BitArray, shape: List(Int), dtype: #(atom.Atom, Int)) {
  let nif_result = from_binary_nif(bin, shape, dtype)
  io.debug("nif result")
  io.debug(nif_result)
  let result = nif_result_to_result(nif_result)
  io.debug(result)
  result
}

@external(erlang, "native", "to_binary")
pub fn to_binary_nif(
  tensor: OrtTensor,
  bits: Int,
  limit: Int,
) -> NifResult(BitArray, String)

pub fn to_binary(
  tensor: OrtTensor,
  bits: Int,
  limit: Int,
) -> Result(BitArray, String) {
  to_binary_nif(tensor, bits, limit) |> nif_result_to_result
}

@external(erlang, "native", "slice")
pub fn slice_nif(
  tensor: OrtTensor,
  start_indicies: List(Int),
  lengths: List(Int),
  strides: List(Int),
) -> Result(OrtTensor, String)

@external(erlang, "native", "reshape")
pub fn reshape_nif(
  tensor: OrtTensor,
  shape: List(Int),
) -> NifResult(OrtTensor, String)

pub fn reshape(tensor: OrtTensor, shape: List(Int)) -> Result(OrtTensor, String) {
  reshape_nif(tensor, shape) |> nif_result_to_result
}

@external(erlang, "native", "concatenate")
pub fn concatenate_nif(
  tensor: OrtTensor,
  dtype: atom.Atom,
  axis: Int,
) -> Result(OrtTensor, String)
