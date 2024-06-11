import gleam/dynamic.{type Dynamic}
import gleam/erlang/atom.{type Atom}
import gleam/option.{type Option}

pub type NifResult(a, b)

pub type Model

pub type Tensor

@external(erlang, "native", "ping")
pub fn ping() -> Nil

@external(erlang, "native", "init_result")
pub fn init(path: String, eps: List(Atom), opt: Int) -> Result(Model, String)

@external(erlang, "native", "run_result")
pub fn run(
  model: Model,
  inputs: List(Tensor),
) -> Result(List(#(Tensor, List(Int), atom.Atom, Int)), String)

@external(erlang, "native", "show_ession_result")
pub fn show_session(
  model: Model,
) -> Result(
  #(
    List(#(String, String, Option(List(Int)))),
    List(#(String, String, Option(List(Int)))),
  ),
  String,
)

@external(erlang, "native", "from_binary_result")
pub fn from_binary(
  bin: BitArray,
  shape: List(Int),
  dtype: #(atom.Atom, Int),
) -> Result(Tensor, String)

@external(erlang, "native", "to_binary_result")
pub fn to_binary(
  tensor: Tensor,
  bits: Int,
  limit: Int,
) -> Result(BitArray, String)

@external(erlang, "native", "slice_result")
pub fn slice(
  tensor: Tensor,
  start_indicies: List(Int),
  lengths: List(Int),
  strides: List(Int),
) -> Result(Tensor, String)

@external(erlang, "native", "reshape_result")
pub fn reshape(tensor: Tensor, shape: List(Int)) -> Result(Tensor, String)

@external(erlang, "native", "concatenate_result")
pub fn concatenate(
  tensor: Tensor,
  dtype: atom.Atom,
  axis: Int,
) -> Result(Tensor, String)
