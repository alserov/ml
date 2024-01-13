package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func main() {
	rawInputs := []float64{
		1, 2, 3, 2.5,
		2, 5, -1, 2,
		-1.5, 2.7, 3.3, -0.8,
	}
	rawWeights := []float64{
		0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}

	biases := mat.NewDense(1, 3, []float64{2, 3, 0.5})
	inputs := mat.NewDense(3, 4, rawInputs)
	weights := mat.NewDense(3, 4, rawWeights)

	weights.CloneFrom(weights.T())

	output := mat.NewDense(3, 3, nil)
	for j := 0; j < 3; j++ {
		for i := 0; i < 3; i++ {
			output.Set(i, j, mat.Dot(inputs.RowView(j), weights.ColView(i))+biases.At(0, i))
		}
	}

	fmt.Println(output)
}
