package com.mantono.kann.ga

data class Neuron(
		val function: (Double) -> Double,
		val weights: Array<Double>,
		val bias: Double,
		val connections: Set<Neuron>)
{
	init
	{
		if(weights.isEmpty())
			throw IllegalArgumentException("Array for weights cannot be empty")
	}

	operator fun invoke(inputs: Array<Double>)
	{
		if(inputs.size != weights.size)
			throw IllegalArgumentException("Perceptron har ${weights.size} but got ${inputs.size} inputs")

		val weightedInputs: Double = inputs.asSequence()
				.mapIndexed { index: Int, input: Double -> weights[index] * input }
				.sum() + bias

		val output = function(weightedInputs)

		connections.forEach { it() }
	}
}