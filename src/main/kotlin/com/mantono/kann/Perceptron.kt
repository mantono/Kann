package com.mantono.kann

data class Perceptron(val weights: Array<Double>)
{
	constructor(numberOfWeights: Int):this(generateWeights(numberOfWeights))
	operator fun get(i: Int): Double = weights[i]
}

fun guess(perceptron: Perceptron, inputs: Array<Float>): Int
{
	val i: Double = inputs.indices.asSequence()
			.map { inputs[it] * perceptron.weights[it] }
			.sum()

	return Math.signum(i).toInt()
}

fun correct(perceptron: Perceptron, trainingData: Array<Float>, error: Int, learningRate: Float): Perceptron
{
	val adjustedWeights: Array<Double> = trainingData.indices.asSequence()
			.map { perceptron[it] + trainingData[it] * error * learningRate }
			.toList()
			.toTypedArray()

	return Perceptron(adjustedWeights)
}

tailrec fun train(per: Perceptron, trainingData: Array<Float>, target: Int, learningRate: Float = 0.1f): Perceptron
{
	val guess: Int = guess(per, trainingData)
	val error: Int = target - guess
	val perceptron = correct(per, trainingData, error, learningRate)
	return if(error == 0) perceptron else train(perceptron, trainingData, target)
}