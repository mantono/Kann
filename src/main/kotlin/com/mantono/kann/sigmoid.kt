package com.mantono.kann

fun sigmoid(x: Double): Double = 1 / (1 + Math.pow(Math.E, -x))