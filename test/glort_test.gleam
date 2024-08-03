import ffi
import gleam/float
import gleam/io
import gleam/list
import gleam/result
import gleeunit
import gleeunit/should
import glort/model as ort_model
import glort/tensor

pub fn main() {
  gleeunit.main()
}

fn arg_max_rec(idx, max_val, max_idx, vals) -> Int {
  case vals {
    [v0, ..vs] if v0 >. max_val -> arg_max_rec(idx + 1, v0, idx, vs)
    [_, ..vs] -> arg_max_rec(idx + 1, max_val, max_idx, vs)
    [] -> max_idx
  }
}

fn arg_max(vals: List(Float)) -> Int {
  case vals {
    [v0, ..vs] -> arg_max_rec(1, v0, 0, vs)
    [] -> -1
  }
}

// gleeunit test functions end in `_test`
pub fn resnet50_test() {
  let assert Ok(input) = tensor.broadcast_float(0.0, 32, [1, 3, 224, 224])
  let assert Ok(model) = ort_model.load("./models/resnet50.onnx")

  let assert Ok([run_output]) = ort_model.run(model, [input])
  let assert Ok(vals) = tensor.to_float_list(run_output, 10_000_000)
  should.equal(arg_max(vals), 499)
}


  pub fn tinymodel_test(){
    let assert Ok(model) = Ortex.load("./models/tinymodel.onnx")
        tensor.
    batch =
      Nx.Batch.stack([
        {Nx.broadcast(0, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(0.0, {100}) |> Nx.as_type(:f32)},
        {Nx.broadcast(1, {100}) |> Nx.as_type(:s32),
         Nx.broadcast(1.0, {100}) |> Nx.as_type(:f32)},
        {Nx.broadcast(2, {100}) |> Nx.as_type(:s32), Nx.broadcast(2.0, {100}) |> Nx.as_type(:f32)}
      ])

    {%Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}, %Nx.Tensor{shape: {3, 10}}} =
      Nx.Serving.run(serving, batch)
  end
}
