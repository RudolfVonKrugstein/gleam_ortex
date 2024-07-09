import ffi.{type OrtTensor}
import gleam/erlang/atom
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string
import glort/dtype.{type Dtype}

@external(erlang, "binary", "copy")
fn copy(data: BitArray, n: Int) -> BitArray

pub type Tensor {
  Tensor(data: OrtTensor, shape: List(Int), dtype: Dtype)
}

pub fn broadcast_float(
  value: Float,
  precision: Int,
  shape: List(Int),
) -> Result(Tensor, String) {
  use t <- result.try(dtype.from_type_and_prec(
    atom.create_from_string("f"),
    precision,
  ))

  let data =
    copy(
      <<value:float-size(precision)>>,
      list.fold(over: shape, from: 1, with: fn(a, b) { a * b }),
    )

  use ort_tensor <- result.try(
    ffi.from_binary(data, shape, #(atom.create_from_string("f"), precision)),
  )

  Ok(Tensor(ort_tensor, shape, t))
}

fn binary_to_floats(
  binary: BitArray,
  precision: Int,
) -> Result(List(Float), String) {
        io.debug(binary)
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

    _ if precision != 32 && precision != 64 -> Error("impossible precision")
    _ -> Error("BitString connaot be converted to float list with precision " |> string.append(int.to_string(precision)))
  }
}

pub fn reshape(tensor: Tensor, shape: List(Int)) -> Result(Tensor, String) {
  let Tensor(ort_tensor, _old_shape, dtype) = tensor

  use reshaped_ort_tensor <- result.try(ffi.reshape(ort_tensor, shape))

  Ok(Tensor(reshaped_ort_tensor, shape, dtype))
}

pub fn flatten(tensor: Tensor) -> Result(Tensor, String) {
  let Tensor(ort_tensor, shape, dtype) = tensor

  case list.length(shape) {
    1 -> {
      Ok(tensor)
    }
    _ ->
      reshape(tensor, [
        list.fold(over: shape, from: 1, with: fn(a, b) { a * b }),
      ])
  }
}

pub fn to_float_list(tensor: Tensor, limit: Int) -> Result(List(Float), String) {
  use Tensor(flattened_ort_tensor, _, t) <- result.try(flatten(tensor))

  let prec = dtype.precision(t)

  use bin <- result.try(ffi.to_binary(flattened_ort_tensor, prec, limit))

  binary_to_floats(bin, prec)
}
