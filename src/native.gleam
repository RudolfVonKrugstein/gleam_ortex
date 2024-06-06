import gleam/dynamic.{type Dynamic}
import gleam/erlang/atom.{type Atom}
import gleam/option.{type Option}
import gleam/result

pub type NifResult(a, b)

pub type Model

pub type Tensor

@external(erlang, "native", "nif_result_to_result")
pub fn nif_result_to_result(nr: NifResult(a, b)) -> Result(a, b)

@external(erlang, "native", "ping")
pub fn ping() -> Nil

@external(erlang, "native", "init")
pub fn init(path: String, eps: List(Atom), opt: Int) -> NifResult(Model, String)

@external(erlang, "native", "run")
pub fn run(
  model: Model,
  inputs: List(Tensor),
) -> NifResult(#(List(Tensor), List(Int), atom.Atom, Int), String)

@external(erlang, "narive", "show_ession")
pub fn show_session(
  model: Model,
) -> Result(
  #(
    List(#(String, String, Option(List(Int)))),
    List(#(String, String, Option(List(Int)))),
  ),
  String,
)

@external(erlang, "native", "from_binary")
pub fn from_binary(
  bin: BitArray,
  shape: List(Int),
  dtype: #(atom.Atom, Int),
) -> NifResult(Tensor, String)

@external(erlang, "native", "to_binary")
pub fn to_binary(
  tensor: Tensor,
  bits: Int,
  limit: Int,
) -> NifResult(BitArray, String)

@external(erlang, "native", "slice")
pub fn slice(
  tensor: Tensor,
  start_indicies: List(Int),
  lengths: List(Int),
  strides: List(Int),
) -> NifResult(Tensor, String)

@external(erlang, "native", "reshape")
pub fn reshape(tensor: Tensor, shape: List(Int)) -> NifResult(Tensor, String)

@external(erlang, "native", "concatenate")
pub fn concatenate(
  tensor: Tensor,
  dtype: atom.Atom,
  axis: Int,
) -> NifResult(Tensor, String)
