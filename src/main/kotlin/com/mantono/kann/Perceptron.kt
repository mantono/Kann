package com.mantono.kann

data class Perceptron(val weights: Array<Double>)
{
	constructor(numberOfWeights: Int):this(generateWeights(numberOfWeights))
}

fun guess(perceptron: Perceptron, inputs: Array<Float>): Int
{
	val i: Double = inputs.indices.asSequence()
			.map { inputs[it] * perceptron.weights[it] }
			.sum()

	return Math.signum(i).toInt()
}

fun correct(per: Perceptron, trainingData: Array<Float>, error: Int): Perceptron
{
	val adjustedWeights: Array<Double> = trainingData.asSequence()
			.map { it * error }
			.map(Float::toDouble)
			.toList()
			.toTypedArray()

	return Perceptron(adjustedWeights)
}

tailrec fun train(per: Perceptron, trainingData: Array<Float>, target: Int): Perceptron
{
	val guess: Int = guess(per, trainingData)
	val error: Int = target - guess
	val perceptron = correct(per, trainingData, error)
	return if(error == 0) perceptron else train(perceptron, trainingData, target)
}