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
  let res = tensor.broadcast_float(0.0, 32, [1, 3, 224, 224])
  //let assert Ok(input) = tensor.broadcast_float(0.0, 32, [1, 3, 224, 224])
  //let assert Ok(model) = ort_model.load("./models/resnet50.onnx")

  //let run_output = ort_model.run(model, [input])
  //io.debug(run_output)
  //let assert Ok([#(res, _shape, _dtype, prec)]) = run_output
  //io.debug(tensor.to_float_list(res, prec, 10_000_000))
}
