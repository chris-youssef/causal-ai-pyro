import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

def exam (p_pass):
    passed = pyro.sample("passed", dist.Bernoulli(p_pass))
    return passed

print(exam(0.7))

def exam_score(average, std_dev):
    score = pyro.sample("score", dist.Normal(average,std_dev))
    return score

print(exam_score(80,5))

def student_result():
    score = exam_score(80,5)
    return score

print(student_result())


def scale_model(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    measurement = pyro.sample("measurement", dist.Normal(weight, 0.75), obs = torch.tensor(9.5))
    return weight
#print(scale_model(8.5))

def scale_guide(guess):
    mean = pyro.param("mean", torch.tensor(8.5))
    std = pyro.param("std", torch.tensor(1.0), constraint=constraints.positive)
    return pyro.sample("weight",dist.Normal(mean, std))

optimizer = pyro.optim.SGD({"lr": 0.01})
svi = pyro.infer.SVI(model = scale_model, guide = scale_guide, optim = optimizer, loss = pyro.infer.Trace_ELBO())

for i in range(1000):
    svi.step(8.5)

print("Learned mean:", pyro.param("mean").item())
print("Learned std:", pyro.param("std").item())

#git add .
#git commit -m "Describe what I changed"
#git push
