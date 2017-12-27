package com.mantono.kann

fun squaredErrorCost(prediction: Double, target: Double): Double
{
	val delta: Double = prediction - target
	return Math.pow(delta, 2.0)
}

fun squaredErrorCost(predictions: Array<Double>, targets: Array<Double>): Double
{
	return transform(predictions, targets, ::squaredErrorCost).sum()
}

fun squaredErrorCost(result: ResultData): Double
{
	return transform(result.predicted, result.target, ::squaredErrorCost).sum()
}

fun sumOfSquaredErrorCost(predictions: Array<Array<Double>>, targets: Array<Array<Double>>): Double
{
	return transform(predictions, targets) { x, y -> transform(x, y, ::squaredErrorCost).sum() }.sum()
}

fun averageOfSquaredErrorCost(predictions: Array<Array<Double>>, targets: Array<Array<Double>>): Double
{
	return sumOfSquaredErrorCost(predictions, targets) / predictions.size
}

fun averageOfSquaredErrorCost(results: Array<ResultData>): Double
{
	return results.map { squaredErrorCost(it.predicted, it.target) }.sum() / results.size
}

fun slope(prediction: Double, target: Double): Double
{
	return 2 * (prediction - target)
}

inline fun <T, reified V> transform(first: Array<T>, second: Array<T>, crossinline transformation: (x: T, y: T) -> V): Array<V>
{
	if(first.size != second.size)
		throw IllegalArgumentException("Size of arrays must match (${first.size} and ${second.size}")

	return first.asSequence()
			.mapIndexed { index, e -> transformation(e, second[index])}
			.toList()
			.toTypedArray()
}