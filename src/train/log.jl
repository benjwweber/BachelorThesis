using DelimitedFiles

function logloss(path, step, iteration, loss, statistics)
    open(path, "a") do io
        writedlm(io, [step iteration loss statistics... time()])
    end
end

function log_pretraining(path, step, iteration, loss, statistics)
    open(path, "a") do io
        writedlm(io, [step iteration loss statistics.dino.global statistics.dino.local statistics.ibot statistics.koleo time()])
    end
end

function initial_log_pretraining(path)
    log_pretraining(path, 0, 0, 0, (; dino=(; var"global"=0, var"local"=0), ibot=0, koleo=0))
end
