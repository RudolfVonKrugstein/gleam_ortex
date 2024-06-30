import gleam/dynamic.{type Dynamic}
import gleam/erlang/atom.{type Atom}
import gleam/option.{type Option}

pub type Model

pub type Tensor

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
) -> NifResult(Model, String)

pub fn init(path: String, eps: List(Atom), opt: Int) -> Result(Model, String) {
  init_nif(path, eps, opt) |> nif_result_to_result
}

@external(erlang, "native_erlang", "run")
pub fn run_nif(
  model: Model,
  inputs: List(Tensor),
) -> NifResult(List(#(Tensor, List(Int), atom.Atom, Int)), String)

pub fn run(
  model: Model,
  inputs: List(Tensor),
) -> Result(List(#(Tensor, List(Int), atom.Atom, Int)), String) {
  run_nif(model, inputs) |> nif_result_to_result
}

@external(erlang, "native", "show_ession")
pub fn show_session_nif(
  model: Model,
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
) -> NifResult(Tensor, String)

pub fn from_binary(bin: BitArray, shape: List(Int), dtype: #(atom.Atom, Int)) {
  from_binary_nif(bin, shape, dtype) |> nif_result_to_result
}

@external(erlang, "native", "to_binary")
pub fn to_binary_nif(
  tensor: Tensor,
  bits: Int,
  limit: Int,
) -> NifResult(BitArray, String)

pub fn to_binary(
  tensor: Tensor,
  bits: Int,
  limit: Int,
) -> Result(BitArray, String) {
  to_binary_nif(tensor, bits, limit) |> nif_result_to_result
}

@external(erlang, "native", "slice")
pub fn slice_nif(
  tensor: Tensor,
  start_indicies: List(Int),
  lengths: List(Int),
  strides: List(Int),
) -> Result(Tensor, String)

@external(erlang, "native", "reshape")
pub fn reshape_nif(tensor: Tensor, shape: List(Int)) -> Result(Tensor, String)

@external(erlang, "native", "concatenate")
pub fn concatenate_nif(
  tensor: Tensor,
  dtype: atom.Atom,
  axis: Int,
) -> Result(Tensor, String)
