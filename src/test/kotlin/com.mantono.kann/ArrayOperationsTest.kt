package com.mantono.kann

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

@Test
fun arrayAdditionTest()
{
	val a = arrayOf(1.0, 2.0, 3.0, 4.0)
	val b = arrayOf(3.0, 4.0, 5.0, 6.0)
	val c = a + b
	val expected = arrayOf(4, 6, 8, 10)
	assertEquals(expected, c)
}

@Test
fun testHadamardProduct()
{
	val a = arrayOf(1.0, 2.0, 3.0, 4.0)
	val b = arrayOf(3.0, 4.0, 5.0, 6.0)
	val c = a x b
	val expected = arrayOf(3.0, 8.0, 15.0, 24.0)
	assertEquals(expected, c)
}


