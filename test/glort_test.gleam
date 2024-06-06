import gleam/io
import gleeunit
import gleeunit/should
import glort/model as ort_model
import glort/tensor
import native

pub fn main() {
  gleeunit.main()
}

// gleeunit test functions end in `_test`
pub fn resnet50_test() {
  native.ping()
  let assert Ok(input) = tensor.broadcast_float(0.0, 32, [1, 3, 224, 224])
  let assert Ok(model) = ort_model.load("./models/resnet50.onnx")
  let res = ort_model.run(model, [input])
  io.debug(res)
}
