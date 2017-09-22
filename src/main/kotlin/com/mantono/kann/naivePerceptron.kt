package com.mantono.kann

fun main(args: Array<String>)
{
	val p = Perceptron(2)
	val guess = guess(p, arrayOf(4.3f, 2.4f))
	println(guess)
}