package main

import "fmt"

func main() {
	inputs := []float32{1, 2, 3, 2.5}

	// weight length = amount of inputs
	// biases length = weights length
	weights := [][]float32{{0.2, 0.8, -0.5, 1}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}}
	biases := []float32{2, 3, 0.5}

	outputs := make([]float32, 0, len(biases))
	for i, b := range biases {
		outputs = append(outputs, countOutput(inputs, weights[i], b))
	}

	fmt.Println(outputs)
}

func countOutput(i, w []float32, b float32) float32 {
	var res float32

	for idx := range i {
		res += i[idx] * w[idx]
	}

	return res + b
}
