package main

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

type LayerDense interface {
	Forward(inputs *mat.Dense)
	Weights() *mat.Dense
	Biases() []float64
	Output() *mat.Dense
}

func NewLayerDense(nInput int, nNeurons int) LayerDense {
	weights := mat.NewDense(nInput, nNeurons, fillWithRandom(nInput*nNeurons))
	return &layerDense{
		weights: weights,
		biases:  make([]float64, nNeurons),
	}
}

type layerDense struct {
	weights *mat.Dense
	biases  []float64

	output *mat.Dense
}

func (ld *layerDense) Output() *mat.Dense {
	return ld.output
}

func (ld *layerDense) Weights() *mat.Dense {
	return ld.weights
}

func (ld *layerDense) Biases() []float64 {
	return ld.biases
}

func (ld *layerDense) Forward(inputs *mat.Dense) {
	dp := countOutput(inputs, ld.weights, ld.biases)
	ld.output = dp
}

func fillWithRandom(size int) []float64 {
	vals := make([]float64, 0, 10)
	for i := 0; i < size; i++ {
		vals = append(vals, rand.Float64()*2-1)
	}
	return vals
}
