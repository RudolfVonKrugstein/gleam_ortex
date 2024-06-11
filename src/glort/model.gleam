import gleam/erlang/atom
import native

pub fn load(path: String) -> Result(native.Model, String) {
  native.init(path, [atom.create_from_string("cpu")], 3)
}

pub fn run(
  model: native.Model,
  input: List(native.Tensor),
) -> Result(List(#(native.Tensor, List(Int), atom.Atom, Int)), String) {
  native.run(model, input)
}
