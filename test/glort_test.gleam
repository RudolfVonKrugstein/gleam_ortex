import ffi
import gleam/io
import gleeunit
import gleeunit/should
import glort/model as ort_model
import glort/tensor

pub fn main() {
  gleeunit.main()
}

// gleeunit test functions end in `_test`
pub fn resnet50_test() {
  let input_result = tensor.broadcast_float(0.0, 32, [1, 3, 224, 224])
  io.debug(input_result)
  let assert Ok(input) = input_result
  let assert Ok(model) = ort_model.load("./models/resnet50.onnx")

  let assert Ok([run_output]) = ort_model.run(model, [input])
  io.debug(tensor.to_float_list(run_output, 10_000_000))
}
