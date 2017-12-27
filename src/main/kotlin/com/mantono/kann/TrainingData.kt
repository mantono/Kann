package com.mantono.kann

data class TrainingData(val input: Array<Double>, val output: Array<Double>)
{
	constructor(input: Array<Double>, output: Double): this(input, arrayOf(output))
	constructor(input: Double, output: Array<Double>): this(arrayOf(input), output)
	constructor(input: Double, output: Double): this(arrayOf(input), arrayOf(output))
}

data class ResultData(val predicted: Array<Double>, val target: Array<Double>)
{
	init
	{
		if(predicted.size != target.size)
			throw IllegalArgumentException("Training data indicates ${target.size} output nodes, but got ${predicted.size} outputs. Check that the amount of output nodes are correct.")
	}

	constructor(predicted: Double, target: Double): this(arrayOf(predicted), arrayOf(target))

	val squaredError: Double = squaredErrorCost(predicted, target)
	val slope: Double = transform(predicted, target) {x, y -> slope(x, y) }.sum()
}