import theano
from theano import tensor as T

a = T.scalar()
b = T.scalar()

c = a * b

print(c, a , b)

mul = theano.function(inputs=[a,b], outputs=c)

multiply = mul(3, 2)

print(multiply)