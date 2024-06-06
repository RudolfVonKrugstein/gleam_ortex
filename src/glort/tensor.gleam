import gleam/erlang/atom
import gleam/list
import native.{type Tensor}

@external(erlang, "binary", "copy")
fn copy(data: BitArray, n: Int) -> BitArray

pub fn broadcast_float(value: Float, precision: Int, shape: List(Int)) -> Result(Tensor, String) {
  let data =
    copy(
      <<value:float>>,
      list.fold(over: shape, from: 1, with: fn(a, b) { a * b }),
    )
  native.from_binary(data, shape, #(atom.create_from_string("f"), precision)) |> native.nif_result_to_result
}
