using ProgressMeter

for epoch in 1:settings.epochs
    @time for (x,y) in loader(train_data, batchsize=settings.batchsize)
        Flux.reset!(net)
        grads = Flux.gradient(model -> let
                ŷ = model(x[  1:196,:])
                ŷ = model(x[197:392,:])
                ŷ = model(x[393:588,:])
                ŷ = model(x[589:end,:])
                Flux.logitcrossentropy(ŷ, y)
            end, net)
        Flux.update!(opt_state, net, grads[1])
    end

    loss, acc, _ = loss_and_accuracy(net, train_data)
    test_loss, test_acc, _ = loss_and_accuracy(net, test_data)
    @info epoch acc test_acc
    nt = (; epoch, loss, acc, test_loss, test_acc)
    push!(train_log, nt)
end