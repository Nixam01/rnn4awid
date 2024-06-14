#=
LossAndAccuracy:
- Julia version: 
- Author: marcin
- Date: 2024-06-05
=#
Flux.reset!(net)
x1, y1 = first(loader(train_data));
y1hat = net(x1[  1:196,:])
y1hat = net(x1[197:392,:])
y1hat = net(x1[393:588,:])
y1hat = net(x1[589:end,:])
@show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

@show loss_and_accuracy(net, train_data);