package com.mantono.kann

class Neuron(
		weights: List<Double> = emptyList(),
		val bias: Double = 0.0,
		val function: (Double) -> Double = ::sigmoid)
{
	private val weights: MutableList<Double> = ArrayList(weights)
	val size: Int get() = weights.size

	constructor(
			connections: Int,
			weightGenerator: Sequence<Double>,
			bias: Double = weightGenerator.first()/2,
			function: (Double) -> Double = ::sigmoid
	): this(weightGenerator.consume(connections), bias, function)

	override fun toString(): String
	{
		val str = StringBuilder("Neuron(")
		if(weights.isNotEmpty())
		{
			str.append("weights = listOf(")
			str.append(weights.joinToString(separator = ", "))
			str.append("), ")
		}
		str.append("bias = $bias,")
		str.append(" function = ::${function.javaClass.kotlin.simpleName}")
		str.append(")")
		return str.toString()
		//return weights.joinToString(prefix = "w: ", separator = "; ") + ", bias $bias"
	}

	operator fun get(i: Int): Double = weights[i]
	operator fun set(i: Int, weight: Double): Neuron
	{
		val newWeights = ArrayList(weights)
		newWeights[i] = weight
		return Neuron(newWeights, bias, function)
	}

	fun weights(): List<Double> = weights.toList()

	fun feedInput(inputs: Array<Double>): Double
	{
		verifySize(inputs.size)

		val weightedInputs: Double = inputs.asSequence()
				.mapIndexed { index: Int, input: Double -> weights[index] * input }
				.sum() + bias

		return function(weightedInputs)
	}

	private fun verifySize(inputs: Int): Boolean
	{
		if(weights.size < inputs)
		{
			val missing = inputs - weights.size
			randomSequence(inputs.toLong() + weights.sum().toRawBits())
					.take(missing)
					.forEach { weights.add(it) }
			return true
		}
		else if(weights.size > inputs)
		{
			val extra = weights.size - inputs
			for(i in 0 until extra)
				weights.removeAt(0)
			return true
		}
		return false
	}

	fun mutate(mutationFactor: Double): Neuron
	{
		val b: Double = Math.abs(mutationFactor)
		val constraints: ClosedFloatingPointRange<Double> = -b .. b

		if(constraints.isEmpty())
			throw IllegalStateException("Invalid range: $constraints")

		val constrainedSequence = randomSequence(weights[weights.lastIndex].toRawBits())
				.map { it * (mutationFactor * 10)}
				.map { it.coerceIn(constraints) }

		val mutations: Array<Double> = constrainedSequence.take(weights.size).toList().toTypedArray()

		val mutatedWeights: MutableList<Double> = weights
				.mapIndexed { index, weight -> weight + mutations[index] }
				.toMutableList()

		val mutatedBias: Double = bias + constrainedSequence.first()
		return Neuron(mutatedWeights, mutatedBias, function)
	}

	override fun hashCode(): Int
	{
		return weights.asSequence()
				.map { it.toRawBits() }
				.reduce { acc, w -> acc xor w }
				.toInt()
	}
}