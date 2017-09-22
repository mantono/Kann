package com.mantono.kann

operator fun Array<Double>.plus(x: Array<out Double>): Array<Double>
{
	return this.indices.asSequence()
			.map { this[it] + x[it] }
			.toList()
			.toTypedArray()
}


/**
 * Matrix product
 */
operator fun Array<Double>.times(x: Array<out Double>): Array<Double>
{
	return this.indices.asSequence()
			.map { TODO("Make som magic here") }
			.toList()
			.toTypedArray()
}

/**
 * Hadamard product
 */
infix fun Array<Double>.x(x: Array<out Double>): Array<Double>
{
	return this.indices.asSequence()
			.map { this[it] * x[it] }
			.toList()
			.toTypedArray()
}