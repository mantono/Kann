package com.mantono.kann

data class Matrix<T>(val width: Int, val height: Int)
{
	val data: MutableList<T> = ArrayList(width*height)

	operator fun get(x: Int, y: Int): T = data[x + (y * width)]
	operator fun set(x: Int, y: Int, e: T): T
	{
		val previous = data[x + (y * width)]
		data[x + (y * width)] = e
		return previous
	}
}