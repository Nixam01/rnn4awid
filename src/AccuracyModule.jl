
"""
# module accuracymodule

- Julia version:
- Author: marcin
- Date: 2024-06-13

# Examples

```jldoctest
julia>
```
"""
module AccuracyModule
    using Statistics: mean

    export loss_and_accuracy

    function softmax(x)
        exp_x = exp.(x .- maximum(x, dims=1))
        return exp_x ./ sum(exp_x, dims=1)
    end

    function get_gradient(predictions, targets)
        probabilities = softmax(predictions)
        return Float32.(probabilities .- targets)
    end

    function cross_entropy_loss_with_gradient(predictions, targets)
        probabilities = softmax(predictions)
        loss = -mean(sum(targets .* log.(probabilities), dims=1))
        gradient = probabilities - targets
        return loss, Float32.(gradient)
    end

    function loss(predictions, targets)
        probabilities = softmax(predictions)
        loss = -mean(sum(targets .* log.(probabilities), dims=1))
        return loss
    end

    function gradient(predictions, targets)
        probabilities = softmax(predictions)
        return probabilities - targets
    end

    function one_cold(encoded)
        return [argmax(vec) for vec in eachcol(encoded)]
    end

    function loss_and_accuracy(ŷ, y)
        loss, grad = cross_entropy_loss_with_gradient(ŷ, y)
        pred_classes = one_cold(ŷ)
        true_classes = one_cold(y)
        acc = round(100 * mean(pred_classes .== true_classes); digits=2)
        return loss, acc, grad
    end

    function loss_acc(prediction, target)
        pred_classes = one_cold(prediction)
        true_classes = one_cold(prediction)
        acc = round(100 * mean(pred_classes .== true_classes); digits=2)
        probability = softmax(prediction)
        loss = -mean(sum(target .* log.(probability), dims=1))
        return loss, acc
    end
end