package main

import (
	"fmt"
	"gorgonia.org/tensor"
)

func main() {
	inputs := tensor.New(tensor.WithBacking([]float32{1, 2, 3, 2.5}))
	weights := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking([]float32{
		0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}))
	biases := tensor.New(tensor.WithBacking([]float32{2, 3, 0.5}))

	dp, err := tensor.Dot(weights, inputs)
	if err != nil {
		panic("failed to find dot product: " + err.Error())
	}

	output, err := tensor.Add(dp, biases)
	if err != nil {
		panic("failed to add: " + err.Error())
	}

	fmt.Println(dp, output)
}
