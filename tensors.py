import torch
import numpy as np

# Examples of Tensors being initialized
data = [[1,2],[3,4]]

# Directly from data, data type is inferred
xData = torch.tensor(data)

# From a NumPy array
npArray = np.array(data)
xNP = torch.from_numpy(npArray)

# From another tensor, retaining properties of the original
xOnes = torch.ones_like(xData)
print(f"Ones Tensor: \n {xOnes} \n")

# From another tensor, overriding the datatype
xRand = torch.rand_like(xData, dtype=torch.float)
print(f"Random Tensor: \n {xRand} \n")

# shape defines the tensor dimensions
shape = (2,3,)
randomTensor = torch.rand(shape)
onesTensor = torch.ones(shape)
zerosTensor = torch.zeros(shape)

print(f"Random Tensor: \n {randomTensor} \n")
print(f"Ones Tensor: \n {onesTensor} \n")
print(f"Zeros Tensor: \n {zerosTensor} \n")

# Tensors have attributes for their shape, datatype, and device
attrDemoTensor = torch.rand(3,4)
print(f"Shape: {attrDemoTensor.shape}")
print(f"Datatype: {attrDemoTensor.dtype}")
print(f"Device: {attrDemoTensor.device} \n")

# Torch has a ton of tensor operations buil-in, apparently similar to NumPy
tensor = torch.ones(4,4)
print(f"{tensor} \n")
tensor[:,1] = 0
print(f"{tensor} \n")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"{t1} \n")

# Operations with a _ suffix are performed in-place
tensor.add_(5)
print(tensor)