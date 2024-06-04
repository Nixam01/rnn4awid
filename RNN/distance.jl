using SymEngine
function f(x::Number,y::Number)
    return x^2 + y^2
end
x, y = symbols("x y")
dfdx = diff(f(x,y), x)
dfdy = diff(f(x,y), y)
delta = [dfdx, dfdy]
subs.(delta, x => 3, y => -2)