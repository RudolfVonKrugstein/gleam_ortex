import gleam/erlang/atom
import gleam/io
import gleam/list
import gleam/result
import native.{type Tensor}

@external(erlang, "binary", "copy")
fn copy(data: BitArray, n: Int) -> BitArray

pub fn broadcast_float(
  value: Float,
  precision: Int,
  shape: List(Int),
) -> Result(Tensor, String) {
  let data =
    copy(
      <<value:float>>,
      list.fold(over: shape, from: 1, with: fn(a, b) { a * b }),
    )
  native.from_binary(data, shape, #(atom.create_from_string("f"), precision))
}


fn binary_to_floats(
  binary: BitArray,
  precision: Int,
) -> Result(List(Float), String) {
  case binary {
    <<>> -> Ok([])
    <<a:float-32, rest:bits>> if precision == 32 -> {
      use tail <- result.try(binary_to_floats(rest, 32))
      Ok([a, ..tail])
    }

    <<a:float-64, rest:bits>> if precision == 64 -> {
      use tail <- result.try(binary_to_floats(rest, 64))
      Ok([a, ..tail])
    }

    _ if precision == 32 || precision == 64 -> Error("impossible precision")
    _ -> Error("BitString connaot be converted to float list")
  }
}

pub fn to_float_list(
  tensor: Tensor,
  precission: Int,
  limit: Int,
) -> Result(List(Float), String) {
  io.debug(tensor)
  use bin <- result.try(native.to_binary(tensor, precission, limit))
  binary_to_floats(bin, precission)
}
