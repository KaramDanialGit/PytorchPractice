import torch

tensor_a = torch.rand(7, 7)
tensor_b = torch.rand(1, 7)
tensor_mult = torch.mm(tensor_a, tensor_b.T)

MANUAL_SEED = 0
torch.manual_seed(MANUAL_SEED)
tensor_a = torch.rand(7, 7)
torch.random.manual_seed(MANUAL_SEED)
tensor_b = torch.rand(1, 7)
tensor_mult = torch.mm(tensor_a, tensor_b.T)

torch.cuda.manual_seed(seed=1234)
torch.manual_seed(1234)
gpu_tens_a = torch.rand(2, 3)
torch.random.manual_seed(1234)
gpu_tens_b = torch.rand(2, 3)

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_tens_a.to(device)
gpu_tens_b.to(device)
gpu_tens_mult = torch.mm(gpu_tens_a, gpu_tens_b.T)

# print(torch.min(gpu_tens_mult))
# print(torch.max(gpu_tens_mult))

# print(gpu_tens_mult.argmin())
# print(gpu_tens_mult.argmax())

torch.random.manual_seed(seed=7)
final_tensor = torch.rand(size=(1, 1, 1, 10))
print(final_tensor)
print(torch.squeeze(final_tensor))