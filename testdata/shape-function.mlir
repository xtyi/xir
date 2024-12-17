func.func private @shape_cal_0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> tensor<2xindex> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %from_elements = tensor.from_elements %dim, %dim_0 : tensor<2xindex>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x?xf32>
  %dim_2 = tensor.dim %arg3, %c0 : tensor<?x?xf32>
  %from_elements_3 = tensor.from_elements %dim_1, %dim_2 : tensor<2xindex>
  %0 = shape.get_extent %from_elements, %c0 : tensor<2xindex>, index -> index
  %1 = shape.get_extent %from_elements_3, %c1 : tensor<2xindex>, index -> index
  %from_elements_4 = tensor.from_elements %0, %1 : tensor<2xindex>
  return %from_elements_4 : tensor<2xindex>
}
