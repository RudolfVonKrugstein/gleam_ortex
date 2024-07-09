import ffi
import gleam/erlang/atom
import gleam/list
import gleam/result
import glort/dtype
import glort/tensor.{type Tensor}

pub type Model {
  Model(ffi.OrtModel)
}

pub fn load(path: String) -> Result(Model, String) {
  use ort_model <- result.try(ffi.init(
    path,
    [atom.create_from_string("cpu")],
    3,
  ))

  Ok(Model(ort_model))
}

pub fn run(model: Model, inputs: List(Tensor)) -> Result(List(Tensor), String) {
  let Model(ort_model) = model

  let inputs = list.map(inputs, fn(input) { input.data })

  use result <- result.try(ffi.run(ort_model, inputs))

  Ok(
    list.map(result, fn(data) {
      let #(ort_tensor, dims, t, prec) = data
      let assert Ok(t) = dtype.from_type_and_prec(t, prec)
      tensor.Tensor(ort_tensor, dims, t)
    }),
  )
}
